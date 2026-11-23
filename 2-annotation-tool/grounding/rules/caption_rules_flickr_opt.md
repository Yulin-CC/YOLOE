# 📖 Caption 组合规则（flickr_opt，原 aerial 改名）

- **概述**: 把已选定的短语组成一句短英文 caption。只改句式，不改短语。动作和方位写在句子里。
- **日期**: 2026-11-16

---

## 1 Caption-核心原则

|要求|说明|
|---|---|
|短语锁定|每条 phrase 必须原样出现（拼写、空格、连字符一致）|
|禁止替换|不得改写成同类词（`a white car` → `car` / `vehicle` / `a car`）|
|禁止增删|不得省略给定短语，也不得另加未给定的目标|
|禁止内容|不要提及标注框、crop、polygon、json、dataset、model|
|短语权重|信息密度优先于华丽修辞|
|短语指代|以给定短语本身做主语/宾语，少用 they/it 把短语消掉|
|句式可变|允许调整语序、介词、连接词；优先一句短句，句末句号|
|句式语言|caption 正文仅用英文|
|实体一致|同一 `obj_id` 的多条短语是同一个物体|
|实体分立|不同 `obj_id` 是不同物体，禁止 is / is also / who is also|

## 2 Caption-风格

|要求|说明|
|---|---|
|动作与方位|用 standing / walking / parked / on / near 写在短语外面，不要改短语|
|不同目标|用 near / beside / and 等分开写|
|同目标多短语|合成一个主体（同位、with）；禁止 near / beside 拆成多个物体|
|句式规范|一条 caption 可含多句；全部给定短语必须用上|

## 3 示例

### 3.1 不同目标

- [obj 0] a white car
- [obj 1] a young man
- [obj 2] a man in a blue shirt

可接受：
> A young man is standing near a white car and a man in a blue shirt.

不可接受：
> A car is near a man. ← 短语被改写

### 3.2 同一目标

- [obj 0] a white car
- [obj 0] a small white car

可接受：
> A small white car, a white car, is parked on the road.

不可接受：
> A white car stands near a small white car. ← 同一车被写成两辆

## 4 输出格式

只返回 JSON：

```json
{"caption": "..."}
```
