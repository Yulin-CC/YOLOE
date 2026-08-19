1. 目的

    a. 基于 [官方LVIS验证集] 以及 [GEOAI无人机场景测试集]，对比测试 [官方模型] 和 [自训模型] 的性能;
    b. 完成 0-QuickStart/_mutilmodel_eval/testlog.md

2. 三个模型的路径

    a. 官方模型：weights/yoloe-11s-seg.pt; (已完成)
    b. 自训模型：runs/0-train/YOLOE-Scratch-260708-v0.1.0/weights/best.pt; (已完成)
    c. 自训模型：runs/0-train/YOLOE-Scratch-260708-v0.1.0-Probe/weights/best.pt

3. 三个验证集的配置文件

    a. LVIS 开集：data/val-yolo_dataset/Public-lvis.yaml;
    b. 智慧安防：data/val-yolo_dataset/GEOAI-Smartsecurity.yaml;
    c. 智慧水务：data/val-yolo_dataset/GEOAI-Smartwaterarea.yaml

注意：

    a. 如果没有的参数留空就行;
    b. 尽量不改动源码，实在需要的话用临时脚本，保持工作区干净
    c. 如果遇到跑不过去的，先询问用户，不擅自对源码做修改
    d. 有 2 个 GPU，应该每个都还有足够的显存，评估一下，可以的话都利用上
