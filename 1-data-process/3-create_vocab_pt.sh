#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-07-07
 # @Description: 离线生成词汇表 JSON 与嵌入 .pt（训练前执行一次即可）
 #   Step 1：grounding cache → global_grounding_neg_cat.json + global_grounding_neg_embeddings.pt
 #   Step 2：yolo yaml + grounding cache → train_label_embeddings.json + train_label_embeddings.pt
###

#---------------#
# 需要修改的值
#---------------#
yolo_dataset="data/yolo/0-YOLO.yaml"                       # YOLO 数据集 yaml
grounding_dataset="data/0-Grounding.yaml"                  # Grounding 数据集 yaml（需已生成 .cache）
#----------------------------------------------------------#
neg_vocab="config/vocab/global_grounding_neg_cat.json"
#----------------------------------------------------------#
vocab_json="config/vocab/train_label_embeddings.json"      # 可选：输出，同步写出词汇表 JSON 备份
#----------------------------------------------------------#
min_freq=100       # 负样本短语最小出现次数
force="--force"    # 强制重建 .pt （可注释掉）
#------------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin

# 脚本在 1-data-process/ 下执行，Python 工具以项目根目录解析相对路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

#---------------#
# Step 1：负样本词表 + 嵌入
#---------------#
python 1-data-process/tools/vocab_generate_global_neg_cat.py \
    --grounding-yaml "$grounding_dataset" \
    --min-freq       "$min_freq"          \
    --neg-vocab      "$neg_vocab"         \
    $force

#---------------#
# Step 2：正样本词表 + 嵌入（scratch 训练必需）
#---------------#
python 1-data-process/tools/vocab_generate_label_embedding.py \
    --yolo-yaml      "$yolo_dataset"      \
    --grounding-yaml "$grounding_dataset" \
    --vocab-json     "$vocab_json"        \
    $force
