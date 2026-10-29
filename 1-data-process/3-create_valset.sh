#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-10-28
 # @LastEditTime: 2026-10-28
 # @Description: jsons-GD → (YOLO)(VAL) 验证集
 #   生成 labels/、val.txt、labels.cache（以及 annotations/lvis_v1_val.json）
###

#---------------#
# 需要修改的值
#---------------#
Path="../testdir/GEOAI-((demo))-Selfcollect-2608-(YOLO)(VAL)"   # 含 images/ 与 jsons-GD/
#---------------#

#---------------#
# 切换到虚拟环境
#---------------#
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate yoloe

#---------------#
# jsons-GD → labels/*.txt + val.txt + labels.cache
#---------------#
python util/create_valset.py --path "$Path"
