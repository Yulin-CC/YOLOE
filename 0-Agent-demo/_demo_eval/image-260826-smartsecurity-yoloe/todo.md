# Agent-autodemo TODO

- **updated**: 2026-10-28 16:26
- **work_root**: 0-Agent-demo/_demo_eval
- **slot**: _demo_eval
- **phase**: confirm
- **current**: 评估包装已建，等待确认是否开评
- **next**: 确认后跑 util/2-eval.sh
- **blocked**: 无

## Meta
- conda: yoloe
- entry: bash 0-Agent-demo/_demo_eval/image-260826-smartsecurity-yoloe/util/2-eval.sh
- customer: Customer.md §202610
- spec: 无
- log_dir: 0-Agent-demo/_demo_eval/image-260826-smartsecurity-yoloe/logs
- results_dir: 0-Agent-demo/_demo_eval/image-260826-smartsecurity-yoloe/0-results

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
| v6 × ValSet | pending | runs/val/geoai/best_GEOAI-ValSet-Smartsecurity | geoai + text |

## Notes
- 入口冻结为票内 `util/2-eval.sh`，转发 `0-QuickStart/scripts/eval.py --mode geoai`
