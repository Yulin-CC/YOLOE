# 📖 目标描述短语规则（Flickr）

- **概述**: 为闭集里的单个目标写 1–3 条短英文名词短语，特异性可以不同。衣服、身体部位不单独成目标。
- **日期**: 2026-10-30

---

## 1 目标描述-规则要求

### 1.1 通用要求

|要求|说明|
|---|---|
|目标描述|为单个裁剪目标生成简短英文名词短语|
|类别锁定|给定 class label 为真值；可换成同类角色或近义（person → a man / a worker），不得串类|
|短语条数|每个目标 **1–3** 条，不足不注水|
|短语形式|名词短语，常见 2–4 个词，最长约 6 个词；可用 a/an/the；不要完整句子|
|描述语言|短语正文仅用英文|
|禁止内容|不要提及裁剪边框、绿色框或标注痕迹|

### 1.2 写法

|要求|说明|
|---|---|
|允许|颜色、年龄、大小、角色、可见穿戴（写进该目标短语）|
|禁止|动作姿态（walking / parked）、方位（on the road / on the left）|
|禁止拆目标|不要单独输出衣物或身体部位（~~a blue shirt~~、~~shaggy hair~~、~~their hands~~）。穿戴写进人：`a man in a blue shirt`|
|只指当前实例|一条短语只描述这一个目标，不要用 two / several / a group of 去概括其它框|
|特异性|同组拉开特异性：`a man` / `a young man` / `a man in a blue shirt`。禁止只换冠词或同义换皮|

### 1.3 例子

|类别|正例|反例|
|---|---|---|
|person|a man, a young man, a man in a blue shirt|a blue shirt, their hands, a man walking, two men|
|car|a car, a white car, a small white car|a parked car, a car on the road, white sedan on the left|

## 2 输出格式

只返回 JSON：

```json
{"phrases": ["a man", "a young man", "a man in a blue shirt"]}
```
