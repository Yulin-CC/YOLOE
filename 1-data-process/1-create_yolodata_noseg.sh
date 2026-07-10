#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-07-09
 # @LastEditTime: 2026-07-09
 # @Description: LabelMe bbox → COCO bbox → SAM segmentation
 #   数据集目录需含 images/ 与 jsons/
###

WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$(cd "$WORK_DIR/.." && pwd)"

#------------------------------------------#
# 需要修改的值
#------------------------------------------#
Path="$PROJECT_ROOT/sample/bbox"           # 数据集根目录（含 images/ 与 jsons/）
#------------------------------------------#
devices="0"                                # SAM 使用的 GPU（单卡 "0"；多卡 "0,1"）
sam_batch=1                                # SAM batch 模式（1=开启，大图/多框建议开）
#------------------------------------------#

output_name="$(basename "$Path")_coco"
bbox_json="${Path}/${output_name}.json"
segm_json="${Path}/${output_name}_segm.json"

cd "$PROJECT_ROOT"

if [ ! -d "${Path}/images" ] || [ ! -d "${Path}/jsons" ]; then
    echo "❌ 目录需包含 images/ 与 jsons/: ${Path}"
    exit 1
fi

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

#---------------#
# Step 1：LabelMe → COCO bbox
#---------------#
echo "Step 1/2  LabelMe → COCO bbox → ${bbox_json}"
conda activate yoloe
python "$WORK_DIR/util/convert_labelme_to_coco_bbox.py" \
    --input  "$Path" \
    --output "$bbox_json"

#---------------#
# Step 2：COCO bbox → SAM segmentation
#---------------#
echo "Step 2/2  SAM → ${segm_json}"
conda activate sam2
_sam_args=(--img-path "${Path}/images" --json-path "$bbox_json" --gpus "$devices")
if [ "$sam_batch" = "1" ] || [ "$sam_batch" = "true" ] || [ "$sam_batch" = "True" ]; then
    _sam_args+=(--batch)
fi
python "$WORK_DIR/tools/generate_sam_masks.py" "${_sam_args[@]}"

echo "✅ 完成  bbox: ${bbox_json}  segm: ${segm_json}"
