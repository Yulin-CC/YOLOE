#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-10-28
 # @LastEditTime: 2026-10-28
 # @Description: 票 image-260826-smartsecurity-yoloe 开集推理入口
 #   薄包装 0-QuickStart/scripts/predict.py，不改仓库源码
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TICKET_DIR="$(cd "$WORK_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORK_DIR/../../../.." && pwd)"
TICKET="${TICKET_DIR#$PROJECT_ROOT/}"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices=0
#------------------------------------------#
dataset="/home/yulin/0-data/2-DroneObject/SmartSecurity/GEOAI-((YOLOE_SmartSecurity-gd01))-Selfcollect-2608-(YOLO)(VAL)/images"
#------------------------------------------#
mode="text"                                # text | visual | promptfree
#------------------------------------------#
project="YOLOE-demo-260826-v6"
#-------------------------------------------------------#
weights="./runs/0-train/$project/weights/best.pt"
mobileclip="./weights/mobileclip_blt.pt"
#-------------------------------------------------------#
names="red car"                            # 仅 text 模式；逗号分隔，类名内可含空格
#-------------------------------------------------------#
config="config/default_notrain.yaml"
#-------------------------------------------------------#


source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行推理程序
#---------------#
cd "$PROJECT_ROOT"

if [ -f "$dataset" ]; then
    output="$(dirname "$dataset")/repro/$project"
elif [ -d "$dataset" ]; then
    output="$dataset/repro/$project"
else
    echo "Error: dataset not found: $dataset"
    exit 1
fi

mkdir -p "$output" "$TICKET_DIR/logs"
export MOBILECLIP_PATH="$mobileclip"
python 0-QuickStart/scripts/predict.py \
    --config   "$config"  \
    --mode     "$mode"    \
    --weights  "$weights" \
    --source   "$dataset" \
    --names    "$names"   \
    --output   "$output"  \
    --device   "cuda:$devices"
echo "✅ 推理完成 → $output"
echo "   vis/crops 在该目录下"
