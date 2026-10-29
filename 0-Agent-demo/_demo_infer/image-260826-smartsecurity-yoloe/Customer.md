## 202610

1. 目的：用 v6 权重在安防 ValSet 图上做文本开集推理（默认问 `red car`）
2. 环境：`conda activate yoloe`
3. 任务：infer
4. 本仓库入口：`bash 0-Agent-demo/_demo_infer/image-260826-smartsecurity-yoloe/util/1-inference.sh`
5. 模型 / ckpt：

    | tag | path |
    | --- | --- |
    | v6 best | `runs/0-train/YOLOE-demo-260826-v6/weights/best.pt` |

6. 数据：`/home/yulin/0-data/2-DroneObject/SmartSecurity/GEOAI-((YOLOE_SmartSecurity-gd01))-Selfcollect-2608-(YOLO)(VAL)/images`
7. 本批规格：`customer/prompts.md`
8. 成功标准：`repro/YOLOE-demo-260826-v6/{vis,crops}/` 写出叠框图
9. 结果：

    a. 文档写回：`0-results/`
    b. 可视化或权重：输入目录下 `repro/$project/`
    c. 日志：`logs/`
