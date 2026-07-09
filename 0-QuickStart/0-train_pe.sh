#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-15
 # @LastEditTime: 2026-06-16
 # @Description: YOLOE PE 训练入口，超参从 config/train_pe.yaml 读取
###
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$(cd "$WORK_DIR/.." && pwd)"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices="0"                                # GPU 设备 ID，多卡如 "0,1,2,3"
#------------------------------------------#
project="YOLOE-2606-test"                  # 权重保存至 runs/0-train/$project/
#------------------------------------------#
model="weights/yoloe-11s-seg.pt"           # .pt → 微调；.yaml → scratch
mobileclip="./weights/mobileclip_blt.pt"   # MobileCLIP 文本编码器权重（text 模式必需）
#------------------------------------------#
dataset="data/yolo/0-YOLO.yaml"            # YOLO 格式数据集 yaml
#------------------------------------------#
config="config/train_pe.yaml"              # 训练配置
#------------------------------------------#
background=1                               # 1=后台运行（nohup）；0=前台盯着看
#------------------------------------------#
# epochs / batch / lr 等超参从 config/train_pe.yaml 中 train.defaults 读取，无需在此修改


#---------------#
# 切换到虚拟环境（conda 路径因机器而异，clone 后请修改 CONDA_BASE）
#   常见：$HOME/miniconda3 | $HOME/anaconda3 | /opt/conda
#   查找：dirname "$(dirname "$(which conda)")"
#   环境名 yoloe 须与 README §0 中 conda create -n 一致
#---------------#
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"   # 本机示例：/home/ubuntu/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate yoloe


#---------------#
# 运行训练程序
#---------------#
cd "$PROJECT_ROOT"

mkdir -p runs/0-train

export MOBILECLIP_PATH="$mobileclip"

# mode 留空时不传 CLI 参数，回退到 config/train_pe.yaml train.defaults.mode
mode_arg="";  [ -n "$mode" ] && mode_arg="--mode $mode"

_run_bg=0
if [ "$background" = "1" ] || [ "$background" = "True" ] || [ "$background" = "true" ]; then
  _run_bg=1
fi

if [ "$_run_bg" = "1" ]; then
  # 日志单独放 logs/，避免预创建 $project 目录导致 Ultralytics 递增到 project2
  log_file="$PROJECT_ROOT/runs/0-train/logs/${project}.log"
  mkdir -p "$(dirname "$log_file")"
  nohup python 0-QuickStart/scripts/train_pe.py \
      $mode_arg          \
      --config  "$config"  \
      --model   "$model"   \
      --data    "$dataset" \
      --project "$project" \
      --device  "$devices" \
      > "$log_file" 2>&1 &
  echo "✅ 后台训练已启动，PID: $!"
  echo "   日志: $log_file"
  echo "   查看: tail -f $log_file"
  echo "   停止: kill $!"
else
  python 0-QuickStart/scripts/train_pe.py \
      $mode_arg          \
      --config  "$config"  \
      --model   "$model"   \
      --data    "$dataset" \
      --project "$project" \
      --device  "$devices"
fi

# ---------------------------------------------------------------
# 大规模预训练三阶段（从头，需准备 Objects365 + Flickr + GQA 数据）
# ---------------------------------------------------------------
# 阶段 1：文本提示词训练（30 epochs，8 卡）
# python z-others/train_seg.py
#
# 阶段 2：视觉提示词（SAVPE 模块，2 epochs）
# python z-others/train_vp.py
#
# 阶段 3：Prompt-Free 嵌入（1 epoch）
# python z-others/train_pe_free.py
