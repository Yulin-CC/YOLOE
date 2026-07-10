# 👁️ YOLOE 训练/推理代码

YOLOE（Real-Time Seeing Anything）—— 开放集目标检测与分割，支持文本 / 视觉 / Prompt-Free 三种推理。

  - 调研笔记参考: https://www.wolai.com/6BYdTivH3Pe7YG6CdDehom


## 更新日志
- [x] 2026-06-16 新增 `0-QuickStart/` 推理、评估、PE 微调、开集训练入口
- [x] 2026-07-02 1. 新增 `1-data-process/` PE 与 Grounding 两套独立数据预处理脚本；2. 新增 `data/yolo/`、`data/grounding/` 训练 yaml 自动生成；3. 开集训练支持 yolo + grounding yaml 合并（scratch 模式）
- [x] 2026-07-07 1. 调整项目结构，合并数据预处理脚本为 `data/create_data.py`; 2. 解耦项目的配置文件和数据读取文件
- [x] 2026-07-09 修复环境安装文档，补全 `z-others/requirements.txt` / `pyproject.toml` 及根目录软链
---

## README 目录

- [0 环境](#0-环境)
- [1 推理（预训练权重）](#1-推理预训练权重)
- [2 评估（复现官方 baseline）](#2-评估复现官方baseline)
- [3 训练 PE 模型（闭集 / 场景迁移）](#3-训练-pe-模型闭集--场景迁移)
- [4 训练开集模型（YOLO + 文本 grounding）](#4-训练开集模型yolo--文本-grounding)
- [5 使用训练后的模型推理](#5-使用训练后的模型推理)
- [6 PE 与 Grounding 对比](#6-pe-与-grounding-对比)
- [附录：数据集（百度云）](#附录-数据集百度云)
- [参考](#参考)

---
## 0 环境

> ⭐ 先搜索项目中的 source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"，将其改成实际的conda路径

- 必装环境

  ```bash
  conda create -n yoloe python=3.10 -y
  conda activate yoloe

  cd /path/to/yoloe-main/
  ln -sf z-others/pyproject.toml pyproject.toml
  ln -sf z-others/requirements.txt requirements.txt
  pip install --upgrade pip setuptools wheel
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 选装环境（SAM2）

  ```bash
    # 标签只有 bbox，没有 segment 的情况下，需要用 sam 来生成
    conda create -n sam2 python==3.10.16
    conda activate sam2
    pip install -r third_party/sam2/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -e third_party/sam2/ -i https://pypi.tuna.tsinghua.edu.cn/simple
    mkdir -p weights
    wget -O weights/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  ```

- 预训练权重下载（推理 / 微调必需）

  ```bash
  mkdir -p weights
  pip install huggingface-hub==0.26.3
  export HF_ENDPOINT=https://hf-mirror.com

  huggingface-cli download jameslahm/yoloe yoloe-11s-seg.pt --local-dir weights/
  huggingface-cli download jameslahm/yoloe yoloe-11s-seg-pf.pt --local-dir weights/
  huggingface-cli download jameslahm/yoloe yoloe-11s-seg-coco.pt --local-dir weights/

  wget -O weights/mobileclip_blt.pt \
    https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
  ```

  - 官网权重：https://huggingface.co/jameslahm/yoloe/tree/main

---

## 1 推理（预训练权重）

- 修改 `0-QuickStart/1-inference.sh` 中的图像路径、文本提示和模型路径

  - **devices**：GPU ID

  - **dataset**：输入图片或目录

  - **mode**：`text` / `visual` / `promptfree`

  - **weights**：预训练 `.pt` 权重

  - **names**：检测类别（仅 text 模式，空格分隔）

  ```bash
  bash 0-QuickStart/1-inference.sh
  ```
## 2 评估（复现官方baseline）

- 2.1 数据准备：

  - 需要准备 LVIS 和 COCO 数据
  - 下载链接：见附录
  - 解压

- 2.2 修改数据路径：

  - LVIS 评估：编辑 `ultralytics/cfg/datasets/lvis.yaml` 中的 `path:` 为你的 LVIS 实际路径
  - COCO 评估：编辑 `ultralytics/cfg/datasets/coco.yaml` 中的 `path:` 为你的 COCO 实际路径

- 2.3 执行下面命令

  ```bash
  bash 0-QuickStart/2-eval.sh  # 修改其中的参数为具体参数，或修改 config/default_open.yaml
  ```
  
---

## 3 训练 PE 模型（闭集 / 场景迁移）

> **PE 流程**：标准 YOLO 分割标注 → 微调已有 `.pt` 权重。**不使用 Grounding 数据。**

### 📁 3.1 PE 训练验证数据集

#### 3.1.1 PE 数据集结构如下所示

- 单数据集结构（标注转换前）

  ```markdown
  ├── path/to/your/dataset          # 数据集路径（前缀 GEOAI-<name>）
  │   ├── images                    # 图像文件 [.jpg/.png]
  │   └── jsons-segment             # LabelMe 分割 json 标注
  ```

- 训练验证集总构建结构（有这个就可以训练了✅）

  ```markdown
  ├── data/0-{project}.yaml           # ⭐训练读取 yaml⭐
  │
  ├── path/to/your/trainvalset        # 训练验证根目录
  │   └── GEOAI-<name>-<date>-YOLO/   # 前缀 GEOAI（不含 -GD 后缀）
  │       ├── images                  # 图像文件
  │       ├── jsons-segment           # 原始标注
  │       ├── labels                  # YOLO seg 标签 [.txt]
  │       ├── train.txt               # 训练索引
  │       └── val.txt                 # 验证索引
  ```

#### 3.1.2 准备自己的 PE 数据集

- 用数据标签处理工具生成 **labels**、**train.txt**、**val.txt**

  - 修改 `1-data-process/1-create_yolodata.sh`

    - **Path**：数据集路径

    - **split_ratio**：训练集比例（默认 0.9）

  - 运行脚本

    ```bash
    cd 1-data-process
    bash 1-create_yolodata.sh
    ```

  - 生成的文件如下所示 

    ```markdown
    ├── path/to/your/dataset          # 数据集路径
    │   ├── jsons-segment             # 原始标注
    │   ├── labels                    # ⭐YOLO seg 标签⭐
    │   ├── train.txt                 # ⭐训练索引⭐
    │   └── val.txt                   # ⭐验证索引⭐
    ```

- 用数据整理工具生成 **训练读取 yaml**

  - 修改 `data/yolo/create_data.py`

    - **path**：数据集根路径列表

    - **project**：项目名称

    - **nc / names**：类别数与类别名

  - 运行脚本，在 `data/yolo/` 下生成 `0-{project}.yaml`

    ```bash
    cd data/yolo
    python create_data.py
    ```

### 3.2 🔧 修改配置文件

- 修改 `0-QuickStart/0-train_pe.sh`

  - **dataset**：YOLO yaml，如 `data/yolo/0-Person.yaml`

  - **model**：预训练 `.pt` 权重

  - **mode**：`linear` / `full` / `visual`（不填则读 yaml 默认值）

  - **project**：实验名称 → `runs/0-train/<project>/`

- 超参默认值：`config/train_pe.yaml`

  - **mode=linear**：仅训练 PE 层（cv3.*.2），推荐 epochs=10

  - **mode=full**：全参微调，推荐 epochs=80

  - **mode=visual**：仅训练 SAVPE 模块，推荐 epochs=2

### 3.3 🚀 开始 PE 训练

  ```bash
  bash 0-QuickStart/0-train_pe.sh
  ```

---

## 4 训练开集模型（YOLO + 文本 grounding）

### 📁 4.1 训练数据集

#### 4.1.1 训练/验证数据集概览 (数据集下载见附录)

- **a. 汇总结构示例**

  > 总共需要三种数据： 【训练】YOLO + Grounding【验证】YOLO

    ```markdown
    ├── path/to/your/trainvalset              # 训练验证根路径
    │   ├── GEOAI-Objects365v1-2607-YOLO      #【YOLO】Objects365v1 ⭐
    │   ├── GEOAI-<name>-<date>-YOLO          #【YOLO】self-dataset 
    │   ├── GEOAI-GQA-2607-GD                 #【Grounding】GQA ⭐
    │   ├── GEOAI-Flickr30k-2607-GD           #【Grounding】GQA ⭐
    │   └── GEOAI-<name>-<date>-GD            #【Grounding】self-dataset

    ```

- **b. 处理前的数据集结构（cache 生成前）**

  ```markdown
  ├── data
  │
  ├── path/GEOAI-Objects365v1-2607-YOLO    #【YOLO】训练根目录: 前缀 GEOAI + 后缀 YOLO
  │   ├── images                           #   图像文件
  │   ├── jsons-segment                    #   标签文件 coco (源数据没有，自己的数据可以按这个格式构建)
  │   └── labels                           #   标签文件yolo
  ├── path/GEOAI-<name>-<date>-YOLO        #【YOLO】训练根目录: 新增的自己的数据集
  ├── ...
  ├── ...  
  ├── GEOAI-GQA-2607-GD                    #【Grounding】训练根目录：前缀 GEOAI + 后缀 GD
  │   ├── images                           #   图像文件
  │   └── jsons                            #   原始 grounding json
  ├── GEOAI-Flickr30k-2607-GD              #【Grounding】训练根目录：前缀 GEOAI + 后缀 GD
  │   ├── images                           #   图像文件
  │   └── jsons                            #   原始 grounding json
  ├── GEOAI-<name>-<date>-GD               #【Grounding】训练根目录：新增的自己的数据集
  ├── ...
  ├── ...
  │
  ├── EVAL-LVIS                            #【YOLO】验证根目录（scratch 默认验证集，YOLO 格式非 Grounding）
  │   ├── images                           #  图像文件
  │   │   └── val2017                      #  COCO val2017 图像
  │   ├── labels                           #  YOLO 标签 [.txt]
  │   │   └── val2017                      #  与 images/val2017 一一对应
  │   ├── annotations                      #  原始 LVIS 标注（评估用）
  │   │   └── `lvis_v1_minival.json`
  │   ├── `minival.txt`                    #  scratch 验证读取文件（约 5000 张）
  │   └── `val.txt`                        #  完整 val 集索引（约 2 万张）
  ```

- **c. 处理后训练集总构建结构（有这个就可以训练了✅）**

  > ⭐为新生成的文件，下面会讲怎么生成（**这一步可跳过，后续再回过头看**）

  ```markdown
  ├── data
  │   ├── `0-Grounding.yaml`               # ⭐Grounding 训练读取 yaml⭐
  │   └── `0-YOLO.yaml`                    # ⭐YOLO 训练读取 yaml⭐
  │
  ├── path/GEOAI-Objects365v1-2607-YOLO    #【YOLO】训练根目录: 前缀 GEOAI + 后缀 YOLO
  │   ├── images                           #   图像文件
  │   ├── jsons-segment                    #   标签文件 coco
  │   ├── labels                           #   标签文件 yolo
  │   └── `train.txt`                      #   ⭐训练读取文件⭐ 
  ├── path/GEOAI-<name>-<date>-YOLO        #【YOLO】训练根目录: 新增的自己的数据集
  ├── ...
  ├── ...  
  ├── path/GEOAI-GQA-2607-GD               #【Grounding】训练根目录：前缀 GEOAI + 后缀 GD
  │   ├── images                           #   图像文件
  │   ├── jsons                            #   原始 grounding json
  │   ├── `gqa_segm.json`                  #   ⭐合并 COCO segm json⭐
  │   └── `gqa_segm.cache`                 #   ⭐训练 cache（实际加载）⭐
  ├── path/GEOAI-Flickr30k-2607-GD              #【Grounding】训练根目录：前缀 GEOAI + 后缀 GD
  │   ├── images                           #   图像文件
  │   ├── jsons                            #   原始 grounding json
  │   ├── `gqa_segm.json`                  #   ⭐合并 COCO segm json⭐
  │   └── `gqa_segm.cache`                 #   ⭐训练 cache（实际加载）⭐
  ├── path/GEOAI-<name>-<date>-GD               #【Grounding】训练根目录：新增的自己的数据集
  ├── ...
  ├── ...
  │
  ├── EVAL-LVIS                            #【YOLO】验证根目录
  │   ├── images                           #  图像文件
  │   │   └── val2017                      #  COCO val2017 图像
  │   ├── labels                           #  YOLO 标签 [.txt]
  │   │   └── val2017                      #  与 images/val2017 一一对应
  │   ├── annotations                      #  原始 LVIS 标注（评估用）
  │   │   └── `lvis_v1_minival.json`
  │   ├── `minival.txt`                    #  scratch 验证读取文件（约 5000 张）
  │   └── `val.txt`                        #  完整 val 集索引（约 2 万张）

  ```

#### 4.1.2 处理数据集

- **训练数据集（COCO）**

  - 对 `path/GEOAI-<name>-<date>-YOLO` 进行单独处理

    ```bash
    cd 1-data-process
    bash 1-create_yolodata.sh          # 注意修改其中参数，只能处理本就带分割标注的标签
    # bash 1-create_yolodata_noseg.sh  # 如果标签只有 bbox，请用该命令 
    ```
    - 转换标签 COCO-segment -> yolo

    - 创建 `train.txt`



  - 最终生成格式

    ```markdown
    ├── path/GEOAI-<name>-YOLO    #【YOLO】训练根目录: 前缀 GEOAI + 后缀 YOLO
    │   ├── images                #   图像文件
    │   ├── jsons-segment         #   标签文件 coco
    │   ├── labels                #   ⭐标签文件 yolo⭐
    │   └── train.txt             #   ⭐训练读取文件⭐ 
    ``` 

- **训练数据集（Grounding）**

  - 对 `path/GEOAI-<name>-<date>-GD` 进行单独处理

    ```bash
    cd 1-data-process
    bash 2-create_grounding.sh  # 注意修改其中参数
    ```
    - 合并标签，生成 `*_segm.json`

    - 进一步生成 `*_segm.cache`

  - 最终生成格式

    ```markdown
    ├── path/GEOAI-<name>-GD    #【Groungding】训练根目录: 前缀 GEOAI + 后缀 GD
    │   ├── images                #   图像文件
    │   ├── jsons-segment         #   标签文件 coco
    │   ├── `gqa_segm.json`                  #   ⭐合并 COCO segm json⭐
    │   └── `gqa_segm.cache`                 #   ⭐训练 cache（实际加载）⭐
    ```
  


- **验证数据集**

  > 修改 `ultralytics/cfg/datasets/lvis.yaml`中的`path`参数成 EVAL-LVIS 的路径


#### 4.1.3 汇总数据集

  - 处理完所有数据集后，进行汇总读取，再次确认数据格式

    - `GEOAI` 前缀为训练数据，`EVAL` 前缀为验证数据

    - `YOLO` 后缀为 YOLO 格式标签数据，`GD` 后缀为 Grounding 格式标签数据

    ```bash
    cd data
    bash create_data.py  # 注意修改其中参数为数据总路径
    ```


### 4.2 🔧 配置文件

- a. 生成开集词汇表

  ```bash
  cd 1-data-process
  bash 3-create_vocab_pt.sh    # 注意修改其中参数
  ```

  - 读取负样本 `*neg_cat.json`，生成 `config/mobileclip:blt/global_grounding_neg_embeddings.pt`

  - 读取训练样本 `0-YOLO.yaml` 和 `0-Grounding.yaml`, 生成 `config/mobileclip:blt/train_label_embeddings.pt`

- b. 修改训练配置 `config/train_open.yaml`

  > 修改如 epochs / batch / lr / imgsz 等参数

### 🚀 4.3 开始开集训练

  - 修改训练配置 `0-QuickStart/0-train_open.sh`

    ```bash
    cd 0-QuickStart
    bash 0-train_open.sh
    ```

---

## 5 使用训练后的模型推理

- 修改 `0-QuickStart/1-inference.sh` 中的模型权重路径

  - **weights**：更换为自己训练后的 `.pt` 文件路径

  ```bash
  bash 0-QuickStart/1-inference.sh
  ```

---

## 6 PE 与 Grounding 对比

|  | PE 微调（§3） | Grounding 开集（§4） |
|--|--------------|---------------------|
| 入口 | `0-train_pe.sh` | `0-train_open.sh` |
| 训练配置 | `config/train_pe.yaml` | `config/train_open.yaml` |
| 推理 / 评估配置 | `config/default_notrain.yaml` | `config/default_notrain.yaml` |
| 数据目录 | `GEOAI-<name>/` 或 `GEOAI-<name>-YOLO/` | YOLO：`GEOAI-<name>-YOLO/`；Grounding：`GEOAI-<name>-GD/` |
| 预处理 | `1-create_yolodata.sh` | YOLO：`1-create_yolodata.sh`；Grounding：`2-create_grounding.sh` |
| yaml 汇总 | 单数据集 yaml（如 `data/yolo/0-Person.yaml`） | `data/create_data.py` → `0-YOLO.yaml` + `0-Grounding.yaml` |
| 词汇表 | 不需要 | `3-create_vocab_pt.sh`（训练前离线生成 `.pt`） |
| 训练读取 | yaml + `train.txt` / `val.txt` | YOLO：yaml + txt；Grounding：`*_segm.json` + `.cache` |
| 验证集 | 数据集内 `val.txt` | `EVAL-LVIS`（YOLO 格式，scratch 默认） |
| 典型模式 | `linear` / `full` / `visual` | `scratch` |
| 典型场景 | 闭集场景迁移 | 开放词汇预训练 |

---

## 附录: 数据集（百度云）

- 训练数据
  - [YOLOE-Obejects365v1](https://pan.baidu.com/s/17QmFqpZXX9SclPK66RD3vg?pwd=wfve)
  - [YOLOE-GQA](https://pan.baidu.com/s/1vtmQlilXQglOXBINLWlj-g?pwd=se9q)
  - [YOLOE-Flickr30K](https://pan.baidu.com/s/16jf3HaefIVzaEbIl1_Expg?pwd=u5rr)
- 验证数据
  - [LVIS](https://pan.baidu.com/s/1AKsXMEFacSO218Svhu3HfQ?pwd=w3vh)
  - [COCO2017](https://pan.baidu.com/s/1idOnx6ZWkfSKCzWkFjTwyg?pwd=j63u)

---
## 参考

- 上游仓库：[THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
- 论文：[YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465)
- 预训练权重：[HuggingFace jameslahm/yoloe](https://huggingface.co/jameslahm/yoloe/tree/main)
- 官方原始文档：`z-others/README.md`

---
# 🎯 Done
