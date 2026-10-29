## 202610

1. 目的：用 v6 权重在安防 ValSet（384 短语）上做开集评估
2. 环境：`conda activate yoloe`
3. 任务：eval
4. 本仓库入口：`bash 0-Agent-demo/_demo_eval/image-260826-smartsecurity-yoloe/util/2-eval.sh`
5. 模型 / ckpt：

    | tag | path |
    | --- | --- |
    | v6 best | `runs/0-train/YOLOE-demo-260826-v6/weights/best.pt` |

6. 数据：`data/val-yolo_dataset/GEOAI-ValSet-Smartsecurity.yaml`（300 图 / 384 短语）
7. 本批规格：无
8. 成功标准：写出 Box/Mask mAP，结果在 `runs/val/geoai/`
9. 结果：

    a. 文档写回：`0-results/`
    b. 可视化或权重：`runs/val/geoai/`
    c. 日志：`logs/`
