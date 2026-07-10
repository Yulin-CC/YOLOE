#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2024-04-17
 # @LastEditTime: 2026-06-16
 # @Description: jsons-segment → YOLO labels + train/val 索引
###

#---------------#
# 需要修改的值
#---------------#
Path="../testdir/GEOAI-person"    # 数据集根目录
split_ratio=0.9                   # 训练集比例
#---------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe

#---------------#
# 标签转换：jsons-segment → labels/*.txt + train.txt / val.txt
#---------------#
python util/create_segment_yolo.py  \
    --path        "$Path"           \
    --split_ratio "$split_ratio"
