#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-08-26
 # @LastEditTime: 2026-08-26
 # @Description: 启动 YOLOE 复核推理页（浏览器上传图片 + 文本提示词）。可信内网使用，无登录。
 # @Command: bash 2-review_server/0-build_server.sh
###
SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$(cd "$SERVER_DIR/.." && pwd)"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
devices=2                                  # GPU 设备 ID → cuda:$devices
#------------------------------------------#
host="0.0.0.0"                             # 监听地址；给他人用必须是 0.0.0.0
port=8088                                  # 浏览器端口
public_ip="113.31.108.24"                  # 公网 IP（发给同事用这个，不是 10.23 内网）
#------------------------------------------#
config="$SERVER_DIR/util/config.json"      # 模型注册表 / 超时 / 上传上限
#------------------------------------------#

# 固定路径（无需修改）
data_dir="$SERVER_DIR/data"
mobileclip="$PROJECT_ROOT/weights/mobileclip_blt.pt"


#---------------#
# 切换到虚拟环境
#---------------#
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yoloe
cd "$PROJECT_ROOT"
export MOBILECLIP_PATH="${MOBILECLIP_PATH:-$mobileclip}"


#---------------#
# 启动复核服务
#---------------#
mkdir -p "$data_dir"
lan_ip=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
if [ -z "$lan_ip" ]; then
  lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
fi

echo "--------------------------------------"
echo "YOLOE review  device=cuda:${devices}  listen=${host}:${port}"
echo "本机访问: http://127.0.0.1:${port}"
if [ "$host" = "127.0.0.1" ] || [ "$host" = "localhost" ]; then
  echo "他人访问: 当前 host=${host}，仅本机可用；请改成 0.0.0.0 后重启"
else
  echo "内网访问: http://${lan_ip:-?}:${port}"
  echo "他人访问: http://${public_ip}:${port}"
fi
echo "config ${config}"
echo "data   ${data_dir}"
echo "--------------------------------------"
python "$SERVER_DIR/util/server.py" \
  --host     "$host" \
  --port     "$port" \
  --device   "cuda:$devices" \
  --config   "$config" \
  --data-dir "$data_dir"
