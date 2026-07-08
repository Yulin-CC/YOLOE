
# 👁️ YOLOE 数据标签结构示意

## 1 Objects365v1

### 1.1 `annotations/objects365_train_segm.json`
> 规模参考：608,606 张图，9,596,485 条标注，365 类。
```json
{
  "categories": [{"id": 1, "name": "person"}, ...],    # 全局类别表；id=Objects365 原始类 id，name=类名字符串（非 yaml 的 0~364 连续索引）
  "images": [
    {
      "id": 367456,                                      # 全局 image_id，annotations.image_id 关联此字段
      "file_name": "obj365_train_....jpg",               # 相对 `images/` 的文件名
      "height": 512,                                     # 图像高（像素）
      "width": 768                                       # 图像宽（像素）
    }
  ],
  "annotations": [
    {
      "id": 0,                                           # 全局 annotation id（实例唯一编号）
      "image_id": 367456,                                # 所属图片 id，对应 images[].id
      "category_id": 1,                                  # 类别 id，查 categories 得类名（如 1→"person"）
      "bbox": [562.16, 192.96, 93.19, 231.20],           # 绝对像素 xywh：[x左上角, y左上角, 宽, 高]
      "segmentation": [[562.16, 192.96, ...]],           # 绝对像素多边形；[x1,y1,x2,y2,...]，一个实例可含多个 polygon
      "iscrowd": 0,                                      # 0=普通实例，1=群体/忽略（训练时通常跳过）
      "area": 21545.79                                   # 实例面积（像素²），segmentation 包围区域
    }
  ]
}
```

---

### 1.2 `jsons-segment/*.json`（LabelMe）

> 由大 JSON **按图拆分**得到，一张图一个文件，stem 与图片名一致。

```json
{
  "version": "0.1.4",                                    # LabelMe 格式版本号
  "flags": {},                                           # 图像级标记（本项目未使用）
  "imagePath": "obj365_train_000000000002.jpg",          # 对应 images/ 下的文件名（非绝对路径）
  "imageData": null,                                     # 内嵌 base64 图片（本项目为空，图片单独存放）
  "imageHeight": 512,                                    # 图像高（像素）
  "imageWidth": 683,                                     # 图像宽（像素）
  "shapes": [                                            # 该图所有实例标注
    {
      "label": "lamp",                                   # 类名字符串（对应 COCO categories[].name）
      "points": [[246.0, 41.0], [242.0, 46.0], ...],     # 多边形顶点，绝对像素 [[x,y], ...]，≥3 个点
      "shape_type": "polygon",                           # 几何类型（本项目均为 polygon）
      "group_id": null,                                  # 实例分组 id（未使用）
      "description": "",                                 # 实例描述（未使用）
      "flags": {},                                       # 实例级标记（未使用）
      "mask": null                                       # 位图 mask（未使用，用 points 多边形代替）
    }
  ]
}
```

## 2 GQA
### 2.1 `gqa_train_segm.json`（合并 COCO Grounding Segm）
> 规模参考：621,143 条 image 记录（含同一物理图的多条 caption），3,662,533 条 annotation。  

```json
{
  "info": [],
  "licenses": [],
  "categories": [],                                      # 空；类名不从 category_id 查表
  "images": [
    {
      "id": 486488,                                      # 全局 image_id（同一物理图的不同 caption 各占一条）
      "file_name": "2354786.jpg",                        # 相对 images/ 的文件名
      "height": 270,                                     # 图像高（像素）
      "width": 500,                                      # 图像宽（像素）
      "original_id": "2354786",                          # GQA 原始图像 id（物理图唯一标识）
      "caption": "two cars on street. a traffic light. ...",  # 该 caption 对应的完整描述句
      "tokens_negative": [[0, 3], [4, 8], ...],           # caption 中负样本/token 区间（字符级 [start, end)）
      "data_source": "vg",                               # 数据来源（如 Visual Genome）
      "dataset_name": "mixed"                            # 混合数据集标记
    }
  ],
  "annotations": [
    {
      "id": 2497319,                                     # 全局 annotation id
      "image_id": 486488,                                # 关联 images[].id（注意：关联 caption 条目，非物理图 id）
      "category_id": 1,                                  # 占位字段（Grounding 不使用，恒为 1）
      "bbox": [189, 132, 166, 80],                       # 绝对像素 xywh：[x左上角, y左上角, 宽, 高]
      "segmentation": [[190.0, 169.0, ...]],             # 绝对像素多边形 [x1,y1,x2,y2,...]
      "tokens_positive": [[0, 3], [4, 8]],                 # 短语在 caption 中的字符区间（可多个 span 拼成一句）
      "iscrowd": 0,                                      # 0=普通实例
      "area": 13280.0                                    # 实例面积（像素²）
    }
  ]
}
```
---
### 2.2 `jsons/*.json`（原始 Grounding，按物理图拆分）
> 由大 JSON **按图拆分**得到，一张图一个文件，stem 与图片名一致。

```json
{
  "info": [],
  "licenses": [],
  "categories": [],                                      # 空
  "images": [
    {
      "id": 0,                                           # 文件内 local id（合并后重排为全局 id）
      "file_name": "107900.jpg",                         # 对应 images/ 下的文件名
      "height": 960,
      "width": 1280,
      "original_id": "107900",                           # 物理图 id
      "caption": "Zebra is bending head down towards ground. short dark mane of a zebra. ...",
      "tokens_negative": [[0, 5], [6, 8], ...],           # 该 caption 的负 token 区间
      "data_source": "vg",
      "dataset_name": "mixed"
    },
    {
      "id": 1,                                           # 同一物理图的第 2 条 caption
      "file_name": "107900.jpg",
      "caption": "Which kind of animal is in the grass? black hair on the zebra's head. ...",
      ...
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,                                     # 关联本文件内 images[].id（非物理图 id）
      "category_id": 1,                                  # 占位，不使用
      "bbox": [271, 70, 572, 877],                       # 绝对像素 xywh
      "segmentation": [[705.0, 147.0, ...]],
      "tokens_positive": [[0, 5]],                       # caption[0:5] → "Zebra"
      "iscrowd": 0,
      "area": 501644.0
    },
    {
      "id": 10,
      "image_id": 1,                                     # 属于第 2 条 caption 的标注
      "tokens_positive": [[14, 20]],                     # → "animal"（在对应 caption 中切片）
      ...
    }
  ]
}
```

---

## 3 Flickr30k

### 3.1 `flickr_train_segm.json`（合并 COCO Grounding Segm）

> 规模参考：148,915 条 image 记录（Flickr 每图 5 句 caption），638,214 条 annotation。

```json
{
  "info": [],
  "licenses": [],
  "categories": [{"id": 1, "name": "object", "supercategory": "object"}],  # 占位 1 类，Grounding 不用
  "images": [
    {
      "id": 0,                                           # 全局 image_id（每句 caption 一条，非每物理图一条）
      "file_name": "3359636318.jpg",                     # 相对 images/ 的文件名
      "height": "334",                                   # 图像高（字符串或整数）
      "width": "500",                                    # 图像宽
      "original_img_id": 3359636318,                     # Flickr 物理图 id
      "sentence_id": 0,                                    # 该图第几句 caption（0~4）
      "caption": "Two people are talking outside of the video game shop ...",  # 单句 caption
      "tokens_negative": [[0, 91]],                      # caption 负 token 区间 [start, end)
      "tokens_positive_eval": [[[0, 10]], [[34, 53]], ...]],  # 评估用正 token（嵌套 span 列表）
      "dataset_name": "flickr"                           # 数据集来源标记
    }
  ],
  "annotations": [
    {
      "id": 0,                                           # 全局 annotation id
      "image_id": 0,                                     # 关联 images[].id（caption 条目）
      "category_id": 1,                                  # 占位，恒为 1
      "bbox": [144.0, 166.0, 64.0, 168.0],               # 绝对像素 xywh
      "segmentation": [[163.0, 168.0, ...]],             # 绝对像素多边形
      "tokens_positive": [[0, 10]],                      # 短语在 caption 中的字符区间 → "Two people"
      "iscrowd": 0,
      "area": 10752.0
    }
  ]
}
```

---

### 3.2 `jsons/*.json`（原始 Grounding，按物理图拆分）

> 一张物理图 1 个 json，通常含 5 条 caption（`sentence_id` 0~4），stem 与图片名一致。

```json
{
  "info": [],
  "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
  "images": [
    {
      "id": 0,                                           # 文件内 local id
      "file_name": "1000092795.jpg",
      "height": "500",
      "width": "333",
      "original_img_id": 1000092795,
      "sentence_id": 0,
      "caption": "Two young guys with shaggy hair look at their hands while hanging out in the yard .",
      "tokens_negative": [[0, 83]],
      "tokens_positive_eval": [[[0, 14]], [[20, 31]], [[40, 51]]],
      "dataset_name": "flickr"
    },
    {
      "id": 1,                                           # 同一物理图第 2 句
      "file_name": "1000092795.jpg",
      "sentence_id": 1,
      "caption": "Two young , White males are outside near many bushes .",
      ...
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,                                     # 关联本文件内 images[].id
      "category_id": 1,
      "bbox": [x, y, w, h],
      "segmentation": [[...]],
      "tokens_positive": [[20, 31]],                     # caption[20:31] → "shaggy hair"
      "iscrowd": 0,
      "area": 1234.0
    }
  ]
}
```

---

## 4 EVAL-LVIS

### 4.1 `annotations/lvis_v1_minival.json`（LVIS 验证标注）

> 规模参考：4,809 张图，50,672 条 annotation，1,203 类。闭集 COCO 格式，无 caption。

```json
{
  "info": {"description": "...", "version": "...", "year": 2018, ...},
  "licenses": [...],
  "categories": [
    {
      "id": 1,                                           # LVIS 类别 id
      "name": "aerosol_can",                             # 类名（下划线格式）
      "synset": "aerosol.n.02",                          # WordNet synset
      "synonyms": ["aerosol_can", "spray_can"],          # 同义词列表
      "def": "a dispenser that holds a substance under pressure",  # 英文定义
      "frequency": "c",                                  # 频率档：r/c/f（rare/common/frequent）
      "instance_count": 11,                              # 训练集实例数
      "image_count": 8                                   # 训练集图像数
    }
  ],
  "images": [
    {
      "id": 397133,                                      # COCO image id
      "file_name": "val2017/000000397133.jpg",           # 相对数据集根目录（含 val2017/ 前缀）
      "height": 427,
      "width": 640,
      "license": 4,
      "flickr_url": "http://farm7.staticflickr.com/...",
      "coco_url": "http://images.cocodataset.org/val2017/...",
      "date_captured": "2013-11-14 17:02:52",
      "neg_category_ids": [279, 899, ...],                 # 该图确定不存在的类 id
      "not_exhaustive_category_ids": [914, 801, ...]     # 该图未穷举标注的类 id
    }
  ],
  "annotations": [
    {
      "id": 1,                                           # annotation id
      "image_id": 446522,                                # 关联 images[].id
      "category_id": 232,                                # 查 categories 得类名
      "bbox": [83.08, 219.88, 301.69, 420.12],           # 绝对像素 xywh
      "segmentation": [[270.75, 598.57, ...]],           # 绝对像素多边形
      "area": 73297.48                                   # 实例面积（像素²）
    }
  ]
}
```

---

## 5 EVAL-COCO2017

### 5.1 `annotations/instances_val2017.json`（COCO val2017 实例分割）

> 规模参考：5,000 张图，36,781 条 annotation，80 类。标准 COCO Instance，无 caption。

```json
{
  "info": {"description": "COCO 2017 Dataset", "version": "1.0", "year": 2017, ...},
  "licenses": [...],
  "categories": [
    {
      "id": 1,                                           # COCO 类别 id（1~90 有缺号）
      "name": "person",                                  # 类名
      "supercategory": "person"                          # 超类
    }
  ],
  "images": [
    {
      "id": 397133,                                      # COCO image id
      "file_name": "000000397133.jpg",                   # 相对 images/val2017/ 的文件名
      "height": 427,
      "width": 640,
      "license": 4,
      "flickr_url": "http://farm7.staticflickr.com/...",
      "coco_url": "http://images.cocodataset.org/val2017/...",
      "date_captured": "2013-11-14 17:02:52"
    }
  ],
  "annotations": [
    {
      "id": 1768,                                        # annotation id
      "image_id": 289343,                                # 关联 images[].id
      "category_id": 18,                                 # 查 categories 得类名（如 dog）
      "bbox": [473.07, 395.93, 38.65, 28.67],            # 绝对像素 xywh
      "segmentation": [[510.66, 423.01, ...]],           # 绝对像素多边形（或 RLE dict）
      "area": 702.11,
      "iscrowd": 0                                       # 0=实例，1=群体
    }
  ]
}
```
