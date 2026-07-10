#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-16
 # @LastEditTime: 2026-07-07
 # @Description: YOLOE 开集训练入口，由 train_open.py 统一完成：
 #   词汇表 .pt 请先运行 1-data-process/3-create_vocab_pt.sh 离线生成
 #   Step 1 开集训练（linear/full/visual/scratch，超参从 config/train_open.yaml 读取）
 #   训练启动时自动备份配置至 runs/0-train/$project/config/{args,dataset,vocab}/
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$(cd "$WORK_DIR/.." && pwd)"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices="0,1"                              # GPU ID；多卡 DDP 如 "0,1,2,3"（batch 为全局 batch，自动按卡数均分）
#------------------------------------------#
project="YOLOE-Scratch-260708-test"          # 权重保存至 runs/0-train/$project/
#--------------------------------------------------------------------#
model="ultralytics/cfg/models/11/yoloe-11-seg.yaml"                  # .pt → 微调；.yaml → scratch
mobileclip="./weights/mobileclip_blt.pt"                             # MobileCLIP 文本编码器权重（必需）
#--------------------------------------------------------------------#
yolo_dataset="data/0-YOLO.yaml"                         # YOLO 格式数据集 yaml
grounding_dataset="data/0-Grounding.yaml"               # Grounding 数据集 yaml（scratch 模式使用）
#-------------------------------------------------------#
config="config/train_open.yaml"                         # 训练配置（开集专用）
#-------------------------------------------------------#
background=1                                            # 1=后台运行（nohup）；0=前台盯着看
#-------------------------------------------------------#
# epochs / batch / lr 等超参从 config/train_open.yaml 中读取，无需在此修改


source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate yoloe


#---------------#
# 运行开集训练程序
#---------------#
cd "$PROJECT_ROOT"

mkdir -p runs/0-train

export MOBILECLIP_PATH="$mobileclip"

# mode 留空时不传 CLI 参数，回退到 config/train_open.yaml train.defaults.mode
mode_arg="";  [ -n "$mode" ] && mode_arg="--mode $mode"

_run_bg=0
if [ "$background" = "1" ] || [ "$background" = "True" ] || [ "$background" = "true" ]; then
  _run_bg=1
fi

if [ "$_run_bg" = "1" ]; then
  # 日志单独放 logs/，避免预创建 $project 目录导致 Ultralytics 递增到 project2
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
