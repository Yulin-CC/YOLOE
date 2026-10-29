# 本票问法

改 `util/1-inference.sh` 的 `names` 后重跑。逗号分隔，类名内可含空格。

| names | 说明 |
| --- | --- |
| `red car` | 复核页默认；v1 ColorCar 用来看颜色短语 |
| `white car` | 对照：v0.2.3 在 0.25 阈值下会全空 |
| `car` | 闭集短词，分通常高于颜色短语 |
| `person wearing yellow vest` | ValSet 短语样例 |
