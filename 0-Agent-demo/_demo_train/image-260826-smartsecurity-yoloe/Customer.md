## 202610

1. 目的：还原并续训 YOLOE-demo-260826-v6（安防开集）
2. 环境：`conda activate yoloe`
3. 任务：train
4. 本仓库入口：`bash 0-Agent-demo/_demo_train/image-260826-smartsecurity-yoloe/util/0-train.sh`
5. 模型 / ckpt：

    | tag | path |
    | --- | --- |
    | 起点 | `weights/yoloe-11s-seg.pt`（盘上待确认） |
    | v6 best | `runs/0-train/YOLOE-demo-260826-v6/weights/best.pt` |

6. 数据：`data/0-YOLO-v6.yaml` + `data/0-Grounding.yaml`（安防 YOLO 全量训练集目前不在盘上）
7. 本批规格：无
8. 成功标准：能按 v6 配方开训，权重写到 `runs/0-train/YOLOE-demo-260826-v6/`
9. 结果：

    a. 文档写回：`0-results/`
    b. 可视化或权重：`runs/0-train/YOLOE-demo-260826-v6/`
    c. 日志：`logs/`
