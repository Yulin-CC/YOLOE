import torch
from torch.nn import functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.torch_utils import smart_inference_mode, select_device
from ultralytics.utils import LOGGER, TQDM, ops
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.model import YOLOEModel
from copy import deepcopy

class YOLOEValidatorMixin:
    @smart_inference_mode()
    def get_visual_pe(self, model):
        assert(isinstance(model, YOLOEModel))
        data_loader, names = self.get_lvis_train_vps_loader(model)
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))
        
        desc = "Get visual prompt embeddings from samples"
        
        for batch in data_loader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count
        
        cls_visual_num = cls_visual_num.to(self.device)
        
        pbar = TQDM(data_loader, total=len(data_loader), \
            desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])
            assert(preds.shape[0] == 1)
            
            cls = batch["cls"].squeeze(-1).to(torch.int).unique(sorted=True)
            assert(len(cls) == 1)
            visual_pe[cls] += preds[0][cls] / cls_visual_num[cls]
        
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)
        
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch

    def get_lvis_train_vps_loader(self, model):
        lvis_train_vps_data =  check_det_dataset('lvis_train_vps.yaml')
        lvis_train_vps_loader = build_dataloader(
            build_yolo_dataset(self.args, lvis_train_vps_data.get('val'), \
                1, lvis_train_vps_data, mode="val", \
                    stride=max(int(model.stride.max()), 32), rect=False, \
                        load_vp=True),
            1,
            self.args.workers,
            shuffle=False,
            rank=-1
        )
        return lvis_train_vps_loader, lvis_train_vps_data["names"]
    
    def add_prefix_for_metric(self, stats, prefix):
        prefix_stats = {}
        for k, v in stats.items():
            if k.startswith("metrics"):
                prefix_stats[f"{prefix}_{k}"] = v
            else:
                prefix_stats[k] = v
        return prefix_stats

    def _resolve_val_prompt_names(self, trainer=None):
        """Resolve text-prompt names for validation.

        Training may keep placeholder names on `trainer.model` / dataset for open-vocab
        checkpoints. Validation still needs the real val-yaml names (e.g. LVIS 1203).
        """
        val_yaml = None
        if trainer is not None:
            data_cfg = getattr(trainer.args, "data", None)
            if isinstance(data_cfg, dict):
                val_cfg = data_cfg.get("val") or {}
                if isinstance(val_cfg, dict):
                    yolo_data = val_cfg.get("yolo_data") or []
                    if yolo_data:
                        val_yaml = yolo_data[0]
            if val_yaml is None:
                # final_eval may point validator.args.data at the val yaml path
                data_arg = getattr(self.args, "data", None)
                if isinstance(data_arg, str):
                    val_yaml = data_arg
        if isinstance(val_yaml, str):
            data = check_det_dataset(val_yaml)
            return [name.split("/")[0] for name in list(data["names"].values())]
        return [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]

    @staticmethod
    def _save_class_state(model):
        return {
            "names": deepcopy(model.names),
            "nc": model.model[-1].nc,
            "pe": getattr(model, "pe", None),
            "has_pe": hasattr(model, "pe"),
        }

    @staticmethod
    def _restore_class_state(model, state):
        """Undo set_classes so EMA/checkpoint does not keep val prompt names."""
        model.names = state["names"]
        model.model[-1].nc = state["nc"]
        if state["has_pe"]:
            model.pe = state["pe"]
        elif hasattr(model, "pe"):
            delattr(model, "pe")
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self.device = trainer.device
            
            model = trainer.ema.ema
            assert(isinstance(model, YOLOEModel))
            assert(not model.training)
            
            names = self._resolve_val_prompt_names(trainer)
            class_state = self._save_class_state(model)
            try:
                if not self.args.load_vp:
                    LOGGER.info("Validate using the text prompt.")
                    LOGGER.info(f"Encoding {len(names)} text prompts...")
                    tpe = model.get_text_pe(names)
                    model.set_classes(names, tpe)
                    tp_stats = super().__call__(trainer, model)
                    tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                    stats = tp_stats
                else:
                    LOGGER.info("Validate using the visual prompt.")
                    self.args.half = False
                    vpe = self.get_visual_pe(model)
                    model.set_classes(names, vpe)
                    vp_stats = super().__call__(trainer, model)
                    vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                    stats = vp_stats
            finally:
                # save_model() persists EMA; restore placeholders so LVIS names are not baked in
                self._restore_class_state(model, class_state)
            
            return stats
        else:
            if isinstance(model, YOLOEModel) and not hasattr(model, "pe"):
                self.device = select_device(self.args.device, self.args.batch)
                
                model.eval().to(self.device)
                data = check_det_dataset(self.args.data)
                names = [name.split("/")[0] for name in list(data["names"].values())]
                
                if not self.args.load_vp:
                    LOGGER.info("Validate using the text prompt.")
                    tpe = model.get_text_pe(names)
                    model.set_classes(names, tpe)
                    tp_stats = super().__call__(trainer, deepcopy(model))
                    tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                    stats = tp_stats
                else:
                    LOGGER.info("Validate using the visual prompt.")
                    self.args.half = False
                    vpe = self.get_visual_pe(model)
                    model.set_classes(names, vpe)
                    vp_stats = super().__call__(trainer, deepcopy(model))
                    vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                    stats = vp_stats

                return stats
            else:    
                return super().__call__(trainer, model)

class YOLOEDetectValidator(YOLOEValidatorMixin, DetectionValidator):
    def postprocess(self, preds):
        """NMS 时传入正确 nc，避免 seg 模型输出的 mask 系数被误算作类别数."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )

class YOLOESegValidator(YOLOEValidatorMixin, SegmentationValidator):
    pass
