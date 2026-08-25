#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-15
 # @LastEditTime: 2026-08-20
 # @Description: YOLOE 推理脚本，支持文本提示词 / 视觉提示词 / Prompt-Free 三种模式
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices=0                                  # GPU 设备 ID
#------------------------------------------#
dataset="/path/to/dataset"          # 输入（单张图片 / 目录）
#------------------------------------------#
mode="text"                                # 推理模式：text | visual | promptfree
#------------------------------------------#
project="YOLOE-demo-260819-v0"
#-------------------------------------------------------#
weights="./runs/0-train/$project/weights/best.pt"       # 预训练权重路径（text/visual 模式通用）
mobileclip="./weights/mobileclip_blt.pt"                # MobileCLIP 文本编码器权重（text 模式必需）
#-------------------------------------------------------#
names="car"                              # 检测类别（仅 text 模式；逗号分隔，类名内可含空格）
#-------------------------------------------------------#  例: names="black car,person,white truck"
config="config/default_notrain.yaml"                    # 推理配置 yaml
#-------------------------------------------------------#


source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行推理程序
#---------------#
cd "$WORK_DIR/.."

# 固定路径（无需修改，cd 后判断确保相对路径正确）
# 单文件输入 → <文件所在目录>/repro/$project/{vis,crops}/
# 目录输入   → <目录>/repro/$project/{vis,crops}/
if [ -f "$dataset" ]; then
    output="$(dirname "$dataset")/repro/$project"
elif [ -d "$dataset" ]; then
    output="$dataset/repro/$project"
else
    echo "Error: dataset not found: $dataset"
    exit 1
fi

mkdir -p "$output"
export MOBILECLIP_PATH="$mobileclip"
python 0-QuickStart/scripts/predict.py \
    --config   "$config"  \
    --mode     "$mode"    \
    --weights  "$weights" \
    --source   "$dataset" \
    --names    "$names"   \
    --output   "$output"  \
    --device   "cuda:$devices"
