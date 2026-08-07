#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-15
 # @LastEditTime: 2026-06-15
 # @Description: YOLOE 评估脚本，支持 lvis（LVIS 开集）/ coco（COCO 下游）/ geoai（无人机开集）三种模式
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#--------------------------------------#
# 需要修改的值（CLI 优先；data/split/batch/ 等见 config eval 段）
#--------------------------------------#
devices=0                              # GPU 设备 ID
#--------------------------------------#
Fixed_AP="True"                        # Fixed AP 评估（max_det=1000 + bbox json + eval_fixed_ap）
val_mode="geoai"                        # lvis | coco | geoai
#--------------------------------------#
prompt="text"                          # text | visual | promptfree（visual/promptfree 当前仅支持 lvis 数据）
#---------------------------------------------#
weights="./weights/yoloe-11s-seg.pt"          # 预训练权重路径
mobileclip="./weights/mobileclip_blt.pt"      # MobileCLIP 权重（text 模式必需）
#---------------------------------------------#
config="config/default_notrain.yaml"          # 评估配置 yaml
#---------------------------------------------#e


source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行评估程序
#---------------#
cd "$WORK_DIR/.."

mode_arg=""; [ -n "$val_mode" ] && mode_arg="--mode $val_mode"
prompt_arg=""; [ -n "$prompt" ] && prompt_arg="--prompt $prompt"
fixed_ap_arg=""; [ -n "$Fixed_AP" ] && fixed_ap_arg="--fixed-ap $Fixed_AP"

python 0-QuickStart/scripts/eval.py \
    --config     "$config"     \
    --device     "$devices"    \
    --weights    "$weights"    \
    --mobileclip "$mobileclip" \
    $mode_arg \
    $prompt_arg \
    $fixed_ap_arg


# 数据集验证数据读取路径：
# lvis 验证数据读取路径：/home/yulin/0-data/0-public/grounding/EVAL-LVIS/annotations/lvis_v1_minival.json
# coco 验证数据读取路径: /home/yulin/0-data/0-public/grounding/EVAL-COCO2017/annotations/instances_val2017.json
# geoai 验证数据读取路径：config eval.datasets.geoai.yaml（默认 data/val-yolo_dataset/GEOAI-Smartsecurity.yaml）