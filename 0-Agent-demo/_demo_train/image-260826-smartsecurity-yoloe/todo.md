# Agent-autodemo TODO

- **updated**: 2026-10-28 16:26
- **work_root**: 0-Agent-demo/_demo_train
- **slot**: _demo_train
- **phase**: confirm
- **current**: 票骨架已建，等待确认是否开训
- **next**: 补安防 YOLO 全量训练集与 yoloe-11s-seg.pt 后再跑 util/0-train.sh
- **blocked**: YOLO 全量训练集、预训练权重、vocab .pt 缺失

## Meta
- conda: yoloe
- entry: bash 0-Agent-demo/_demo_train/image-260826-smartsecurity-yoloe/util/0-train.sh
- customer: Customer.md §202610
- spec: 无
- log_dir: 0-Agent-demo/_demo_train/image-260826-smartsecurity-yoloe/logs
- results_dir: 0-Agent-demo/_demo_train/image-260826-smartsecurity-yoloe/0-results

## Checklist
- [x] §0.1 工作目录门禁（构建）
- [x] §2 确认 / 填写 Customer.md
- [x] §5 发现并冻结本仓库入口
- [ ] §4 检查环境与产物路径
- [ ] §4 执行（按单元一行）
- [ ] §4 写回 0-results
- [ ] §4 交付摘要

## Runs
| id | status | artifact | note |
| --- | --- | --- | --- |
| v6-restore | pending | runs/0-train/YOLOE-demo-260826-v6/weights/best.pt | 历史权重已在；续训缺训练集 |

## Notes
- 入口冻结为票内 `util/0-train.sh`，转发 `0-QuickStart/scripts/train_open.py`
- 新票复制本目录，改名为 `image-YYMMDD-场景-功能`
