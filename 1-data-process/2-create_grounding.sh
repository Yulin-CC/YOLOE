#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-16
 # @LastEditTime: 2026-10-28
 # @Description: jsons → train_segm.json + train_segm.cache
 #   训练时 Grounding yaml 只需 img_path，固定读取这两个文件名
###

#---------------#
# 需要修改的值
#---------------#
Path="../testdir/qga"              # 数据集根目录（含 images/ 与 jsons/）
#---------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate yoloe

#---------------#
# Step 1：jsons/*.json → train_segm.json
#---------------#
python util/create_grounding.py \
    --input   "$Path" \
    --output  "${Path}/train_segm.json"

#---------------#
# Step 2：train_segm.json → train_segm.cache（供 GroundingDataset 训练）
#---------------#
python tools/generate_grounding_cache.py \
    --json-path "${Path}/train_segm.json" \
    --img-path  "${Path}/images"
