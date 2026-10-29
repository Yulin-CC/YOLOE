#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-10-28
 # @LastEditTime: 2026-10-28
 # @Description: 票 image-260826-smartsecurity-yoloe 开集评估入口
 #   薄包装 0-QuickStart/scripts/eval.py，不改仓库源码
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TICKET_DIR="$(cd "$WORK_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORK_DIR/../../../.." && pwd)"
TICKET="${TICKET_DIR#$PROJECT_ROOT/}"

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
devices=0
#--------------------------------------#
Fixed_AP="False"                       # geoai 开集用标准 Box/Mask AP；True=Fixed AP
val_mode="geoai"                       # lvis | coco | geoai
#--------------------------------------#
prompt="text"                          # text | visual | promptfree
#---------------------------------------------#
project="YOLOE-demo-260826-v6"
weights="./runs/0-train/$project/weights/best.pt"
mobileclip="./weights/mobileclip_blt.pt"
#---------------------------------------------#
dataset="data/val-yolo_dataset/GEOAI-ValSet-Smartsecurity.yaml"
#---------------------------------------------#
config="config/default_notrain.yaml"
#---------------------------------------------#
background=0


source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行评估程序
#---------------#
cd "$PROJECT_ROOT"

export MOBILECLIP_PATH="$mobileclip"
mkdir -p "$TICKET_DIR/logs"

mode_arg=""; [ -n "$val_mode" ] && mode_arg="--mode $val_mode"
prompt_arg=""; [ -n "$prompt" ] && prompt_arg="--prompt $prompt"
fixed_ap_arg=""; [ -n "$Fixed_AP" ] && fixed_ap_arg="--fixed-ap $Fixed_AP"

if [ "$background" = "1" ]; then
  log_file="$TICKET_DIR/logs/eval.log"
  nohup python 0-QuickStart/scripts/eval.py \
      --config     "$config"     \
      --device     "$devices"    \
      --weights    "$weights"    \
      --mobileclip "$mobileclip" \
      --data       "$dataset"    \
      $mode_arg \
      $prompt_arg \
      $fixed_ap_arg \
      > "$log_file" 2>&1 &
  echo "✅ 后台评估已启动，PID: $!"
  echo "   日志: $log_file"
else
  python 0-QuickStart/scripts/eval.py \
      --config     "$config"     \
      --device     "$devices"    \
      --weights    "$weights"    \
      --mobileclip "$mobileclip" \
      --data       "$dataset"    \
      $mode_arg \
      $prompt_arg \
      $fixed_ap_arg
fi
