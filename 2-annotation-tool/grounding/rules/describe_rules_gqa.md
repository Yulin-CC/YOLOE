# 📖 目标描述短语规则（航拍开集 gqa）

- **概述**: 仿公开 GQA 的短语形态：短名词短语、多数不带 a/an/the，仍可组合。给后续 gd04 对比 aerial（带冠词）用。现在不要跑生成。
- **日期**: 2026-11-17

---

## 1 目标描述-规则要求

### 1.1 通用要求

|要求|说明|
|---|---|
|目标描述|为单个裁剪目标生成简短英文名词短语|
|类别锁定|给定 class label 为真值；人可换成 man / worker，不得串类|
|短语条数|每个目标 **1–3** 条，不足不注水|
|短语形式|名词短语，常见 1–4 个词，最长约 6 个词；**不要 a/an/the**；不要完整句子|
|描述语言|短语正文仅用英文|
|禁止内容|不要提及裁剪边框、绿色框或标注痕迹|

### 1.2 写法

|要求|说明|
|---|---|
|允许|颜色、年龄、大小、角色、可见穿戴（写进该目标短语）|
|禁止冠词|不要 `a white car` / `a man`。写成 `white car` / `man`。模型若带冠词，落地时剥掉|
|禁止|动作姿态（walking / parked）、方位（on the road / on the left）|
|禁止拆目标|不要单独输出衣物或身体部位。穿戴写进人：`man in a blue shirt`（介词后的 a 可以留）|
|只指当前实例|一条短语只描述这一个目标，不要用 two / several / a group of|
|特异性|同组拉开特异性。禁止只换同义词|
|禁止裸类名|禁止单独输出闭集类名：~~car~~、~~truck~~、~~person~~、~~boat~~、~~excavator~~。车辆必须带外观或角色：`white car`、`small white car`|
|禁止车辆子类当中心词|不要用 sedan / SUV / pickup 替换 car。用 car/truck 作中心词，靠颜色、大小组合|

### 1.3 例子

|类别|正例|反例|
|---|---|---|
|person|man, young man, man in a blue shirt|person, a man, a person, blue shirt, man walking, two men|
|car|white car, small white car, dark car|car, a car, a white car, white sedan, parked car|
|truck|white truck, small truck|truck, a truck, parked truck|
|boat|white boat, small boat|boat, a boat|
|excavator|yellow excavator, small excavator|excavator, an excavator|

## 2 输出格式

只返回 JSON：

```json
{"phrases": ["white car", "small white car"]}
```
