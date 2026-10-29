#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-08-23
 # @LastEditTime: 2026-10-28
 # @Description: 票 image-260826-smartsecurity-yoloe 开集训练入口（不改 config/train_open.yaml）
 #   新票复制本目录，改名为 image-YYMMDD-场景-功能，互不覆盖
 #   实际训练仍走 0-QuickStart/scripts/train_open.py
 #   v6：安防 YOLO 100% + Flickr/GQA/安防 GD，权重 runs/0-train/YOLOE-demo-260826-v6/
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TICKET_DIR="$(cd "$WORK_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORK_DIR/../../../.." && pwd)"
TICKET="${TICKET_DIR#$PROJECT_ROOT/}"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices="0,1,2,3"                          # v6 原配方 4 卡；batch 为全局 batch，DDP 按卡数均分
#------------------------------------------#
project="YOLOE-demo-260826-v6"             # 新票务必改名，否则会覆盖 runs/0-train/$project/
#--------------------------------------------------------------------#
model="weights/yoloe-11s-seg.pt"
mobileclip="./weights/mobileclip_blt.pt"
#--------------------------------------------------------------------#
# v1/v3: 0-YOLO.yaml（probe-10 × 25%）
# v6:    0-YOLO-v6.yaml（安防 100%）
yolo_dataset="$TICKET/data/0-YOLO-v6.yaml"
grounding_dataset="$TICKET/data/0-Grounding.yaml"
#-------------------------------------------------------#
config="$TICKET/data/train_colorcar.yaml"
#-------------------------------------------------------#
background=1                                            # 1=后台运行（nohup）；0=前台盯着看
#-------------------------------------------------------#


source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行开集训练程序
#---------------#
cd "$PROJECT_ROOT"

mkdir -p runs/0-train

export MOBILECLIP_PATH="$mobileclip"

# mode 留空时不传 CLI 参数，回退到 train_colorcar.yaml train.defaults.mode
mode_arg="";  [ -n "$mode" ] && mode_arg="--mode $mode"

_run_bg=0
if [ "$background" = "1" ] || [ "$background" = "True" ] || [ "$background" = "true" ]; then
  _run_bg=1
fi

if [ "$_run_bg" = "1" ]; then
  log_file="$PROJECT_ROOT/runs/0-train/logs/${project}.log"
  mkdir -p "$(dirname "$log_file")"
  nohup python 0-QuickStart/scripts/train_open.py \
      $mode_arg                    \
      --config         "$config"   \
      --model          "$model"    \
      --data           "$yolo_dataset"       \
      --grounding-data "$grounding_dataset"  \
      --project        "$project"  \
      --device         "$devices"  \
      > "$log_file" 2>&1 &
  echo "✅ 后台训练已启动，PID: $!"
  echo "   日志: $log_file"
  echo "   查看: tail -f $log_file"
  echo "   停止: kill $!"
else
  python 0-QuickStart/scripts/train_open.py \
      $mode_arg                    \
      --config         "$config"   \
      --model          "$model"    \
      --data           "$yolo_dataset"       \
      --grounding-data "$grounding_dataset"  \
      --project        "$project"  \
      --device         "$devices"
fi
