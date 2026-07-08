#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-15
 # @LastEditTime: 2026-06-15
 # @Description: YOLOE 评估脚本，支持 LVIS zero-shot 评估与 COCO 下游迁移评估
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#--------------------------------------#
# 需要修改的值（CLI 优先；data/split/batch/ 等见 config eval 段）
#--------------------------------------#
devices=0                              # GPU 设备 ID
#--------------------------------------#
Fixed_AP="True"                        # Fixed AP 评估（max_det=1000 + bbox json + eval_fixed_ap）
mode="text"                            # text | visual | promptfree | coco（留空则用 yaml）
#---------------------------------------------#
weights="./weights/yoloe-11s-seg.pt"          # 预训练权重路径
mobileclip="./weights/mobileclip_blt.pt"      # MobileCLIP 权重（text 模式必需）
#---------------------------------------------#
config="config/default_notrain.yaml"          # 评估配置 yaml
#---------------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行评估程序
#---------------#
cd "$WORK_DIR/.."

mode_arg=""; [ -n "$mode" ] && mode_arg="--mode $mode"
fixed_ap_arg=""; [ -n "$Fixed_AP" ] && fixed_ap_arg="--fixed-ap $Fixed_AP"

python 0-QuickStart/scripts/eval.py \
    --config     "$config"     \
    --device     "$devices"    \
    --weights    "$weights"    \
    --mobileclip "$mobileclip" \
    $mode_arg \
    $fixed_ap_arg

# 数据集验证数据读取路径：
# lvis 验证数据读取路径：/home/yulin/0-data/0-public/grounding/EVAL-LVIS/annotations/lvis_v1_minival.json
# coco 验证数据读取路径: /home/yulin/0-data/0-public/grounding/EVAL-COCO2017/annotations/instances_val2017.json