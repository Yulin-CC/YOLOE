# 📖 目标描述短语规则（航拍开集 aerial）

- **概述**: Flickr 结构（1–3 条、有粗有细、可组合），但禁止用闭集类名加冠词当短语。给航拍 YOLOE 开集用。
- **日期**: 2026-11-16

---

## 1 目标描述-规则要求

### 1.1 通用要求

|要求|说明|
|---|---|
|目标描述|为单个裁剪目标生成简短英文名词短语|
|类别锁定|给定 class label 为真值；人可换成 a man / a worker，不得串类|
|短语条数|每个目标 **1–3** 条，不足不注水|
|短语形式|名词短语，常见 2–4 个词，最长约 6 个词；可用 a/an/the；不要完整句子|
|描述语言|短语正文仅用英文|
|禁止内容|不要提及裁剪边框、绿色框或标注痕迹|

### 1.2 写法

|要求|说明|
|---|---|
|允许|颜色、年龄、大小、角色、可见穿戴（写进该目标短语）|
|禁止|动作姿态（walking / parked）、方位（on the road / on the left）|
|禁止拆目标|不要单独输出衣物或身体部位。穿戴写进人：`a man in a blue shirt`|
|只指当前实例|一条短语只描述这一个目标，不要用 two / several / a group of|
|特异性|同组拉开特异性。禁止只换冠词或同义换皮|
|禁止裸类名|禁止单独输出闭集类名加冠词：~~a car~~、~~a truck~~、~~a person~~、~~a boat~~、~~an excavator~~。车辆必须带外观或角色：`a white car`、`a small white car`|
|禁止车辆子类当中心词|不要用 sedan / SUV / pickup 替换 car。用 car/truck 作中心词，靠颜色、大小组合|

### 1.3 例子

|类别|正例|反例|
|---|---|---|
|person|a man, a young man, a man in a blue shirt|a person, a blue shirt, a man walking, two men|
|car|a white car, a small white car, a dark car|a car, a white sedan, a parked car, white sedan on the left|
|truck|a white truck, a small truck|a truck, a parked truck|
|boat|a white boat, a small boat|a boat|
|excavator|a yellow excavator, a small excavator|an excavator|

## 2 输出格式

只返回 JSON：

```json
{"phrases": ["a white car", "a small white car"]}
```
