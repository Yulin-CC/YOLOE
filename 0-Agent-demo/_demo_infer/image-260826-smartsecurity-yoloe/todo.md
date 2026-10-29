# Agent-autodemo TODO

- **updated**: 2026-10-28 16:26
- **work_root**: 0-Agent-demo/_demo_infer
- **slot**: _demo_infer
- **phase**: confirm
- **current**: 推理包装已建，等待确认是否开推
- **next**: 确认后跑 util/1-inference.sh
- **blocked**: 无

## Meta
- conda: yoloe
- entry: bash 0-Agent-demo/_demo_infer/image-260826-smartsecurity-yoloe/util/1-inference.sh
- customer: Customer.md §202610
- spec: customer/prompts.md
- log_dir: 0-Agent-demo/_demo_infer/image-260826-smartsecurity-yoloe/logs
- results_dir: 0-Agent-demo/_demo_infer/image-260826-smartsecurity-yoloe/0-results

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
| v6 × ValSet × red car | pending | images/repro/YOLOE-demo-260826-v6 | text 模式 |

## Notes
- 入口冻结为票内 `util/1-inference.sh`，转发 `0-QuickStart/scripts/predict.py`
- 问法改 `util/1-inference.sh` 的 `names`，或看 `customer/prompts.md`
