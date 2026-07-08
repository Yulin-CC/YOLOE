# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """scratch 混合数据集构建较慢，DDP 下仅 rank0 先建 cache。"""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        from ultralytics.data import build_dataloader

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            from ultralytics.utils import LOGGER
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)
