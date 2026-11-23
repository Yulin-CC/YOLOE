---
name: yoloe-agent-demo
description: >-
  Frozen ops card for the YOLOE repo: env, three dataset layouts, script map,
  train ticket, eval and infer. Read before training, eval, infer, or dataset
  prep in 0-Agent-demo. Use when the user says YOLOE, 开集训练, 智慧安防训练, or
  names this repo.
---

# 🧪 YOLOE Agent 操作卡

  - **概述**: 在本仓做开集训练、验证、推理或数据处理时先读本文件。已冻结的项直接做，只在 [§6] 停下来问。
  - **日期**: 2026-11-23

---

## 0 已冻结

  - **主机 / 仓根 / 解释器**: 由 `workenv/link.md` 与派工给出，卡上不写死。换机器先改那份 link。
  - **conda**: `source <miniconda3>/etc/profile.d/conda.sh && conda activate yoloe`
  - **驱动**: CUDA 12.8。PyPI 上直接 `pip install torch` 会装到 CUDA 13，这台机器不能用。要重装时用 `https://download.pytorch.org/whl/cu128` 或已验证过的 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/`。
  - **卡**: 当前 `0,1` 两张 RTX 4090。开跑前看 `nvidia-smi`。被 vLLM 占着就停，不要杀进程。
  - **新数据路径**: 仍要用户给，禁止猜盘。

---

## 1 环境

  环境已在。重装时在仓根执行：

    ```bash
    conda create -n yoloe python=3.10 -y
    conda activate yoloe
    ln -sf z-others/pyproject.toml pyproject.toml
    ln -sf z-others/requirements.txt requirements.txt
    pip install --upgrade pip setuptools wheel
    pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

  - **SAM2**: 选装。只在标签只有框、要补 polygon 时建 `sam2`（Python 3.10.16），装 `third_party/sam2`，权重 `weights/sam2.1_hiera_large.pt`。
  - **权重**: 只认仓内 `weights/`。现有 `weights/yoloe-11s-seg.pt`、`weights/mobileclip_blt.pt`。缺文件先停，不默认下载。
  - 脚本里的 `source /path/to/miniconda3/etc/profile.d/conda.sh` 一律换成上面的真实路径。

---

## 2 数据三种

  训练要 YOLO 闭集 + Grounding，验证是 YOLO 格式的开集短语集。

  | 角色 | 必备 | 训练怎么读 |
  | --- | --- | --- |
  | YOLO 训练 | `images/`、`labels/`、`train.txt` | 子 yaml 的 `path` + `train: train.txt` |
  | Grounding | `images/`，同级 `train_segm.json`、`train_segm.cache` | yaml 只写 `img_path`，文件名读死 |
  | YOLOE val | `images/`、`labels/`、`val.txt`、`annotations/lvis_v1_val.json` | yaml `names` 与 label 的 class id 对齐（0 起） |

  - `val.txt` 行是 `./images/xxx.jpg`。label 行是 `cls x1 y1 ...`。`lvis` 里 `category_id = cls + 1`。
  - 目录名不要混用。服务器 `1-data-process`：Grounding 源是 `jsons/`，验证源是 `jsons-GD/`。产线仓 `.cursor/tools/1-DataEngine/0-Data-Organize`：开集源是 `jsons-GD_yoloe/`。同一份数据只用一套，不要改名去迁就另一套。
  - 命名：`GEOAI-((Name))-(Public|Selfcollect)-YYMM-(YOLO|GD)`。验证集以 `(YOLO)(VAL)` 结尾。
  - `train_segm.cache` 里的 `im_file` 是生成时的绝对路径。目录改名后，json 和图片还在，缓存仍指向旧路径。训练读的是缓存，不是 yaml 里的新 `img_path`。
  - `annotations/lvis_v1_val.json` 的每张图还要有 `neg_category_ids` 和 `not_exhaustive_category_ids`。`is_lvis` 为 true 时，官方 AP 读这两个字段。缺了会打 `lvis unable to run`，结果表留下的是 ultralytics 内部 mAP，没有 APr、APc、APf。2026-11-15 v1.0 的框 mAP50 约 0.10 是内部指标。demo6 的 0.536 是 LVIS AP50。
  - 源标注没有确认负样本时，`neg_category_ids` 写 `[]`。`not_exhaustive_category_ids` 写这张图没有正样本的 category id（json 里从 1 起，与 categories 顺序一致）。两个都写成空列表时，图上没出现的类不会记成误检。
  - 新验证集由 `create_valset.py` 的 `write_lvis_json` 生成，会带上这两个字段。已经在盘上、并且不是这个函数原样导出的 json，不要整份重跑。gd02 那份图片 id 从 1 起，area 是多边形面积，类别带 `image_count`。函数本身的图片 id 从 0 起，area 用 bbox 宽乘高，frequency 按实例数。缺字段就在原文件上补，id、area、frequency 和条数保持不动。

---

## 3 脚本对照

  数据已在本机、目录符合 [§2] 的服务器读法时，用仓内 `1-data-process/`。目录是 `jsons-GD_yoloe/` 时，用产线三支脚本，工作目录是产线仓的 `tools/1-DataEngine/0-Data-Organize/`。

  | 做什么 | 入口 |
  | --- | --- |
  | YOLO 训练，已有分割 | `1-data-process/1-create_yolodata.sh` |
  | YOLO 训练，只有框 | `1-data-process/1-create_yolodata_noseg.sh` |
  | Grounding：`jsons/` → `train_segm.json` + cache | `1-data-process/2-create_grounding.sh` |
  | 验证：`jsons-GD/` → labels、`val.txt`、lvis json | `1-data-process/3-create_valset.sh`（调 `util/create_valset.py`） |
  | 开训前词表 pt | `1-data-process/4-create_vocab_pt.sh` |
  | 只有框时补分割 | `1-data-process/tools/generate_sam_masks.py`，环境 `sam2` |
  | 扫 `(YOLO)` / `(GD)` 生成索引。不扫 `(VAL)` | `data/create_data.py` |
  | `jsons-GD_yoloe/` → `*_segm.json` | 产线 `util/convert_json2coco.py` |
  | LabelMe caption → `*_segm.json` | 产线 `util/convert_geoai2coco.py` |
  | `train_segm.json` → `.cache` | 产线 `util/grounding_generate_cache.py`（要 ultralytics） |

  改 sh 里的数据路径再跑。不要改这些脚本的转换逻辑。`2-create_grounding.sh`、`3-create_valset.sh`、`4-create_vocab_pt.sh` 的 conda 是 `/path/to/miniconda3`，数据路径是 `../testdir/...`，不要原样执行。缓存和词表按 [§4]。新验证集在仓根用 `yoloe` 调 `util/create_valset.py --path <验证根>`。已有 json 只缺两个图片字段时，按 [§2] 补，不要重跑。

---

## 4 训练

  不改 `0-QuickStart/`、仓库 `config/`、仓库 `data/`。一轮写在 `0-Agent-demo/_demo_train/image-YYMMDD-<场景>-yoloe/`。票内有 `Customer.md`、`todo.md`、`config/default.yaml`、`data/0-YOLO.yaml`、`data/0-Grounding.yaml`、`util/0-train.sh`。不建 `customer/`，不写 `data/train_colorcar.yaml`。

  开训前四样都要在，缺一件就停。词表和缓存的做法在本节后面。

  - YOLO 训练：`labels/` 和 `train.txt`。
  - Grounding：每份都有 `train_segm.json` 和 `train_segm.cache`。抽一条 `im_file`，必须落在现目录且文件存在。
  - 验证：`labels/`、`val.txt`、`annotations/lvis_v1_val.json`。json 每张图有 [§2] 的两个字段。yaml 的 `names` 和 label class id 对齐。
  - 词表两份都在，且不是 0 字节、不是 `.partial`：`config/mobileclip:blt/global_grounding_neg_embeddings.pt`、`config/mobileclip:blt/train_label_embeddings.pt`。训练数据的内容或图片目录变了才重做一次。没变就用现成的。

  读不出的图（全 0、空文件、文件头不是图）移出训练目录，放到同一数据根的 `_cache_bad/`，清单写入该票 `todo.md`。只重做受影响的那份 `train_segm.cache`。不整目录删除。2026-11-14 v1.0 的 gd02 有一张全 0 图，缓存重建前训练在 epoch 0 退出。

  - `util/0-train.sh` 转发 `0-QuickStart/scripts/train_open.py`。`--config` 指票内 `config/default.yaml`，`--data` 指 `data/0-YOLO.yaml`，`--grounding-data` 指 `data/0-Grounding.yaml`。
  - `train_open.py` 先读 `0-YOLO.yaml` 的 `val`。索引里有 val，就不用 `config/default.yaml` 的 `val_data`。
  - 词表在这轮训练数据换了内容，或图片目录改了路径时，重做一次。不按 epoch 重做，也不复用上一张票或 `runs/` 里的 `.pt`。内容和路径都没变就不要重做。
  - 不要直接跑 `1-data-process/4-create_vocab_pt.sh`。里面的 conda 是 `/path/to/miniconda3` 占位，`yolo_dataset` / `grounding_dataset` 指向仓库 `data/`，不是本票。不要改这份 sh 的转换逻辑。
  - 在仓根用 `yoloe` 调两个已有工具，yaml 用本票的。`--min-freq` 读那份 sh 里的现行值（当前 50）。加上 `--force`。不要传 `--output-dir`。`text_model` 来自 `ultralytics/cfg/default.yaml`，当前是 `mobileclip:blt`，所以 `.pt` 写到 `config/mobileclip:blt/`（目录名带冒号）。JSON 备份在 `config/vocab/`。
  - 先负样本，再正样本。解释器用 `yoloe` 环境里的 python。`MOBILECLIP_PATH` 指仓内 `weights/mobileclip_blt.pt`。`PYTHONPATH` 指仓根。

    ```bash
    TICKET=0-Agent-demo/_demo_train/<票>
    export MOBILECLIP_PATH="$PWD/weights/mobileclip_blt.pt"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    python 1-data-process/tools/vocab_generate_global_neg_cat.py \
        --grounding-yaml "$TICKET/data/0-Grounding.yaml" \
        --min-freq 50 \
        --neg-vocab config/vocab/global_grounding_neg_cat.json \
        --force
    python 1-data-process/tools/vocab_generate_label_embedding.py \
        --yolo-yaml "$TICKET/data/0-YOLO.yaml" \
        --grounding-yaml "$TICKET/data/0-Grounding.yaml" \
        --vocab-json config/vocab/train_label_embeddings.json \
        --force
    ```

  - 开训前两个文件都要在，且不是 0 字节、不是 `.partial`：`config/mobileclip:blt/global_grounding_neg_embeddings.pt`、`config/mobileclip:blt/train_label_embeddings.pt`。2026-11-14 v1.0 缺第二份，标签扫完后 `FileNotFoundError` 退出，没有进入 epoch。
  - 票内 `0-YOLO.yaml` 的子 yaml 写成从仓根起的路径。词表工具只按仓根解析相对路径。训练脚本先按索引所在目录找，找不到再按仓根找。`./train-yolo_dataset/...` 这种短路径训练能读，词表工具会报文件不存在。
  - 图片目录改名或搬走时，路径变了的那几份 Grounding 重做 `train_segm.cache`。路径没变的不要重做。不要用软链接把旧目录名指到新目录。
  - 不要直接跑 `1-data-process/2-create_grounding.sh`。里面的 conda 是 `/path/to/miniconda3` 占位，`Path` 是 `../testdir/qga`。json 已在、只是目录改了名时，不要重跑 json 转换。不要改这份 sh 的转换逻辑。
  - 在仓根用 `yoloe` 调 `1-data-process/tools/generate_grounding_cache.py`。`--json-path` 指该数据集现目录下的 `train_segm.json`，`--img-path` 指同目录的 `images/`。`PYTHONPATH` 指仓根。缓存写在 json 旁边，覆盖旧的 `train_segm.cache`。
  - 做完抽一条 `im_file`。它必须落在当前目录里，并且文件存在。还是旧目录名就停，不要开训。2026-11-14 v1.0 的 GQA / Flickr30k 缓存仍指向改名前的旧目录。正式目录是 `GEOAI-((GQA))-Public-2607-(GD)` 和 `GEOAI-((Flickr30k))-Public-2607-(GD)`。训练在 epoch 0 报 `Image Not Found`。gd02 的 `im_file` 已是现目录，那次没有重做。

    ```bash
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    python 1-data-process/tools/generate_grounding_cache.py \
        --json-path "<现目录>/train_segm.json" \
        --img-path "<现目录>/images"
    ```

  - `batch` 是全局 batch，按卡数分。换卡数时先看显存，不要默默改超参。
  - 工程名写进 `--project`，目录落在 `runs/0-train/<工程名>/`。已有同名目录就换名，不覆盖。
  - 参照票：`0-Agent-demo/_demo_train/image-261113-smartsecurity-yoloe-v1/`。新版本另开票。
  - 多卡（`device` 里多于一张卡）会进 DDP。`train_open.py` 只在本进程把仓根插入 `sys.path`，不写 `PYTHONPATH`。`yoloe` 环境里没有安装 `ultralytics` 包，父进程靠这份 `sys.path` 才能 import 仓内包。
  - DDP 临时脚本写到 `~/.config/Ultralytics/DDP/_temp_*.py`，再用 `python -m torch.distributed.run` 另起解释器。子进程的路径是临时目录，看不到仓根，于是 `No module named ultralytics`。2026-11-13 v1.0（PID 3613178，`device=0,1`）在打印完模型结构后这样退出，没有进入 epoch。
  - `ultralytics/utils/dist.py` 的 `generate_ddp_command` 在拉起子进程前会删掉当次 `save_dir`。所以上面这次没有留下 `runs/0-train/YOLOE-SmartSecurity-261113-v1.0/`。日志还在 `runs/0-train/logs/YOLOE-SmartSecurity-261113-v1.0.log`。
  - 多卡开跑前，先确认 DDP 子进程能 import 仓内 `ultralytics`。没确认就停。

---

## 5 评估和推理

  新模型评三套，都用本轮 `best.pt`：**开集 LVIS**（gd02 val）、**闭集终验 val**、**闭集终验 test**。包装写在 `_demo_eval/<票>/util/`，转发 `0-QuickStart/scripts/eval.py`。不要原样跑 `0-QuickStart/2-eval.sh`：conda 是占位，权重还指着 `weights/yoloe-11s-seg.pt`。不改 `0-QuickStart/` 和 `ultralytics`。`prompt=text` 时要仓内 `weights/mobileclip_blt.pt`，`PYTHONPATH` 指仓根。看图走 `0-QuickStart/1-inference.sh`，权重和卡号写在 `_demo_infer` 的票里。这三套出指标。另外有一套固定的可视化测试集，只出图不出指标，换模型也照着跑、不另起问法，见本节末「可视化测试集（图推）」。

  - **输出目录会撞名**：`eval.py` 的 `make_save_dir` 用「权重文件名 stem + '_' + 数据文件名 stem」，落在 `runs/val/geoai/`。自训版权重都叫 `best.pt`，数据 yaml 同名就互相覆盖。每版每套各写一份带版本后缀的 data yaml（如 `closed-val-v1.3.yaml`），产出目录才是 `best_closed-val-v1.3`。开跑前确认目标目录不存在；别的版本已有目录不删、不覆盖。
  - **可视化落点**：`vis/`、`crops/` 写到该数据集根下的 `repro/<模型名>-<yymmdd>/`，不堆在票的 `0-results/`。票 `0-results/` 只放 `testlog.md` 和指标。

  **开集 LVIS**（训练日志里那一列，demo6 的框 mAP50 0.536 是这套）

  - 数据是本票验证 yaml。demo6 是 `GEOAI-ValSet-Smartsecurity.yaml`：300 张、384 个短语。v1.0 是 gd02：1820 张、1738 个短语，`annotations/lvis_v1_val.json` 每张图要有 [§2] 的两个字段。
  - 入口：`python 0-QuickStart/scripts/eval.py --mode geoai --prompt text --fixed-ap false --split val`。验证配置里 `is_lvis: true`，才会出 LVIS 的 AP、AP50、APr、APc、APf。缺字段会打 `lvis unable to run`，表上只剩内部 mAP。
  - 已跑通的包装：`0-Agent-demo/_demo_train/image-261113-smartsecurity-yoloe-v1/util/0-eval.sh`，配置 `util/eval-gd02.yaml`。2026-11-16 v1.0 `best.pt` 的框 AP 0.368、AP50 0.453、APr 0.376、APc 0.300、APf 0.366。掩膜 AP 0.002。json 里 20090 条标注有 20083 条 polygon 多于 4 个点。预测写进 LVIS 的掩膜 RLE 尺寸是 360×640，框坐标是原图像素，两边坐标系不一致，这次掩膜 AP 不能用。
  - 短语数量和 demo6 的 384 不同，不要把 0.453 直接当成没达到 0.536。

  **闭集终验 val**（表上 v2 到 v6 那一列，v6 框 mAP50 0.622 是这套）

  - 23 个类，id 顺序跟 `data/train-yolo_dataset/Selfcollect-smartsecurity.yaml` 的 `names` 一致，从 `person` 到 `drone`。
  - 验证：`/path/to/data/SmartSecurity/GEOAI-((YOLOE_SmartSecurity_valset))-Selfcollect-2608-(YOLO)`，`val.txt` 18204 行。
  - 同一条 `eval.py`，`--mode geoai --prompt text --fixed-ap false --split val`。不要开 `is_lvis`。yaml 写在本票 `data/`，`names` 用上面 23 个，`val: val.txt`。
  - 指标是框和掩膜的 P、R、mAP50、mAP50-95。这套没有 APr、APc、APf。同一次评测两列都出，因为模型同时给框和掩膜。
  - 2026-11-16 v1.0 `best.pt`：验证集框 P 0.721、R 0.594、mAP50 0.639、mAP50-95 0.482，掩膜 P 0.701、R 0.573、mAP50 0.611、mAP50-95 0.382。表上 v6 的 0.622 靠近这次验证集，不靠近测试集。摘要在 `0-Agent-demo/_demo_eval/image-261116-smartsecurity-yoloe-v1/0-results/closed-set-v1.0.md`。

  **闭集终验 test**

  - 同 23 类，同一份 yaml 换 `test: test.txt`，`--split test`。
  - 测试：`/path/to/data/SmartSecurity/GEOAI-((YOLOE_SmartSecurity_testset))-Selfcollect-2608-(YOLO)`，`test.txt` 18204 行。表头写过 2.1 万，盘上现在各是 18204。
  - 2026-11-16 v1.0 `best.pt`：测试集框 P 0.252、R 0.185、mAP50 0.204、mAP50-95 0.158，掩膜 P 0.239、R 0.172、mAP50 0.186、mAP50-95 0.119。测试集普遍低于验证集，别拿 val 的数当 test。

  **可视化测试集（图推）**

  - 固定两套，换任何模型都按本节的提示词和阈值跑，不另起一套问法：
    `/path/to/data/TestSet/GEOAI-SmartSecurity-2608-100-a`
    和 `...-100-b`，各 105 张纯图（2021–2026 分年抽样 100 张 + 5 张 `0-public/sample` 遗留样本），不带标签。
  - 每套自带三份文件，跑前先读，不要覆盖：
    `prompts.txt`（两臂标准提示词）、`index.csv`（逐张年份、场景、原路径、`gt_classes`）、`README.md`（本节口径的完整说明）。
  - 提示词 23 句，顺序与闭集终验的 23 类一致（`person` 打头、`drone` 收尾），框可以对着闭集指标读。
    两臂只差冠词：`noart` 不带、`art` 带。`fire`、`smoke` 是物质名词，两臂写法相同，避免把冠词效应和词形变化混在一起。
    内部代号换成自然英文（`floatingstuff` → `floating object`、`burnmarks` → `burn mark`、`dronenest` → `drone nest` 等），因为文本提示走 MobileCLIP 编码。
  - 入口与闭集同一条 `predict.py`，`--mode text`，`conf 0.1`、`iou 0.7`、`imgsz 640`。
    `--names` 只取 `prompts.txt` 里某一臂那一整串；文件带 `#` 注释行，不要整文件 `readlines()` 塞进去。
  - 输出落数据集侧：`<数据集根>/repro/<模型>-<版本>-<臂>/`，臂取 `noart` 或 `art`，如 `yoloe-v1.4-noart`。
    里面 `vis/` 是叠框整图，`crops/<原图名>/<类名>_<置信度>.jpg`（按原图分文件夹，不是按类分）。
    票 `0-results/` 只放框数对照表，不放图。
  - 每套图的类别覆盖不同，有的类在某套里是空的（如 `-a` 的 `firetruck`、`forklift`、`piledriver`、`towercrane`、`drillingrig`、`heightrestrictionbar`）。
    空的那几类只能看有没有误检成它，看不出召回，报告里要写明，不要当成漏检。
  - 这批图来自 valset，v1.3 / v1.4 挑 best epoch 时用过。**看趋势可以，不能当独立测试成绩报**，也不要拿它替代开集和闭集两套指标。
  - `index.csv` 的 `gt_classes` 只作人工读图参考，不是训练标签。`prompts.txt` 是标准问法，要改先问，不要为了某个模型单独改一份。

---

## 6 停

  下面这些写进该票 `todo.md` 再问，其余按上面做：

  - 权重、`train_segm.cache` 或词表 `.pt` 不在。训练数据换了内容或改了路径，但词表或对应缓存没重做，同样停。缓存里的 `im_file` 还指向旧目录名，也停
  - 验证 yaml 的 `names` 和 label class id 对不上
  - 验证 json 的图片缺 `neg_category_ids` 或 `not_exhaustive_category_ids`。不要为此整份重跑一份已经不是 `write_lvis_json` 导出的 json
  - 开集要 LVIS 的 AP，但验证配置没开 `is_lvis`
  - 闭集终验的 23 个 `names` 和 `data/train-yolo_dataset/Selfcollect-smartsecurity.yaml` 对不上
  - `0,1` 被别的进程占着
  - 新工程名会盖住已有 `runs/0-train/`
  - 用户没给新的数据根目录
  - 多卡时 DDP 子进程 import 不到仓内 `ultralytics`（见 [§4]）
