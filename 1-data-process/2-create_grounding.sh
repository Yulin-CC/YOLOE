#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-16
 # @Description: jsons → COCO segm json + .cache
###

#---------------#
# 需要修改的值
#---------------#
Path="../testdir/qga"              # 数据集根目录（含 images/ 与 jsons/）
project="gqa"                      # 输出文件前缀（生成 {Path}/{project}_segm.json）
#---------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yulin

#---------------#
# Step 1：jsons/*.json → COCO segm json
#---------------#
python util/create_grounding.py \
    --input   "$Path"     \
    --project "$project"  \
    --output  "${Path}/${project}_segm.json"

#---------------#
# Step 2：COCO segm json → .cache（供 GroundingDataset 训练）
#---------------#
python tools/generate_grounding_cache.py       \
    --json-path "${Path}/${project}_segm.json" \
    --img-path  "${Path}/images"
