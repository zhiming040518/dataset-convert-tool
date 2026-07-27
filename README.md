# dstool — 数据集格式转换工具库

**dstool** 是一个轻量级的 Python 命令行工具，用于在不同标注格式之间转换计算机视觉数据集。

支持的标注形状：矩形框（rectangle）、多边形（polygon）、圆形（circle）  
支持的格式转换：

```
LabelMe JSON  ──→  Pascal VOC XML   (json2voc)
LabelMe JSON  ──→  YOLO txt         (json2yolo)
Pascal VOC    ──→  YOLO txt         (voc2yolo)
LabelMe JSON  ──→  Pixel Mask PNG   (json2mask)
```

---

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [命令详解](#命令详解)
  - [dstool json2voc](#dstool-json2voc)
  - [dstool json2yolo](#dstool-json2yolo)
  - [dstool voc2yolo](#dstool-voc2yolo)
  - [dstool json2mask](#dstool-json2mask)
- [LabelMe JSON 输入格式](#labelme-json-输入格式)
- [输出格式说明](#输出格式说明)
- [使用场景示例](#使用场景示例)
- [依赖](#依赖)
- [常见问题](#常见问题)
- [License](#license)

---

## 安装

### 方式一：pip 安装（推荐）

```bash
pip install dstool/
```

安装时会自动下载所需依赖：`opencv-python`、`pillow`、`numpy`、`tqdm`。

### 方式二：开发模式安装

适合需要修改源码的场景：

```bash
git clone <repo-url> dstool
cd dstool
pip install -e .
```

### 环境要求

- Python >= 3.8
- 支持 Windows / Linux / macOS
- 支持 Python 虚拟环境、Anaconda/Miniconda 环境

---

## 快速开始

### 查看帮助

```bash
dstool -h                 # 查看所有可用命令
dstool json2voc -h        # 查看某个命令的详细帮助
```

### 基本用法

dstool 支持两种使用方式：**交互模式**（无参数，按提示操作）和**命令行模式**（直接传参）。

```bash
# 命令行模式（推荐自动化脚本使用）
dstool json2voc  -src ./labels   -output ./VOCdevkit
dstool json2yolo -src ./labels   -output ./YOLO_dataset
dstool voc2yolo  -src ./VOC2007  -output ./YOLO_dataset
dstool json2mask -src ./labels   -output ./masks

# 交互模式（适合不熟悉命令行的用户）
dstool json2voc
# > 请输入包含JSON文件的路径 (留空使用当前目录):
# > 请输入输出路径 (留空使用当前目录\VOCdevkit):
```

---

## 命令详解

### dstool json2voc

将 LabelMe 格式的 JSON 标注文件转换为 Pascal VOC 格式的 XML 标注文件。

```bash
dstool json2voc -src ./json_labels -output ./VOCdevkit
# 别名
dstool json2VOC -src ./json_labels -output ./VOCdevkit
```

**转换逻辑**：

| Shape 类型 | 处理方式 |
|-----------|---------|
| `rectangle` | 取两对角点作为 bbox |
| `polygon` | 取多边形所有顶点的 min/max x, y 作为 bbox |
| `circle` | 取圆心 ± 半径作为 bbox |

**输出结构**：

```
VOCdevkit/
├── Annotations/          # XML 标注文件，每张图片一个 .xml
├── JPEGImages/           # 从源目录复制的图片（如果存在）
└── ImageSets/
    └── Main/
        └── train.txt     # 所有图片文件名列表（不含后缀）
```

**生成的 XML 示例**：

```xml
<annotation>
  <folder>VOC2007</folder>
  <filename>image001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>500</ymax>
    </bndbox>
  </object>
</annotation>
```

**终端输出示例**：

```
源路径:   /home/user/labels
输出路径: /home/user/VOCdevkit

找到 150 个 JSON 文件
转换 JSON→VOC: 100%|██████████| 150/150 [00:02<00:00]
  复制了 120 张图片到 /home/user/VOCdevkit/JPEGImages

[OK] 转换完成！数据集已保存至: /home/user/VOCdevkit
  共处理 150 个JSON文件，3246 个标注对象
  类别: bicycle, car, dog, person
```

---

### dstool json2yolo

将 LabelMe 格式的 JSON 标注文件转换为 YOLO 格式的 txt 标注文件。

```bash
dstool json2yolo -src ./json_labels -output ./YOLO_dataset
```

**转换逻辑**：

- 自动收集所有类别并按字母排序，分配 class_id（0, 1, 2, ...）
- bbox 坐标归一化到 [0, 1] 范围
- 每张图片生成一个同名的 .txt 文件

**归一化公式**：

```
cx = (xmin + xmax) / 2 / image_width
cy = (ymin + ymax) / 2 / image_height
w  = (xmax - xmin) / image_width
h  = (ymax - ymin) / image_height
```

**输出结构**：

```
YOLO_dataset/
├── labels/               # YOLO 标注文件，每张图片一个 .txt
├── classes.txt           # 类别名称列表，一行一个
└── dataset.yaml          # YOLOv5/v8 训练配置文件
```

**生成的 YOLO txt 示例** (`image001.txt`)：

```
0 0.166667 0.437500 0.166667 0.375000
1 0.366667 0.437500 0.066667 0.125000
```

每行格式：`class_id cx cy w h`（全部归一化到 0~1）

**生成的 dataset.yaml**：

```yaml
path: /home/user/YOLO_dataset
train: images
val: images

nc: 4
names: ['bicycle', 'car', 'dog', 'person']
```

---

### dstool voc2yolo

将 Pascal VOC 格式的 XML 标注文件转换为 YOLO 格式的 txt 标注文件。

```bash
dstool voc2yolo -src ./VOCdevkit -output ./YOLO_dataset
# 别名
dstool VOC2yolo -src ./VOCdevkit -output ./YOLO_dataset
```

**源目录查找策略**：

1. 优先在 `Annotations/` 子目录下查找 XML（VOC 标准结构）
2. 如果未找到，递归搜索整个源目录

这意味着你可以直接传入 VOC 根目录：

```bash
# 这两种写法效果相同（如果 VOCdevkit 下有 Annotations/ 目录）
dstool voc2yolo -src ./VOCdevkit -output ./yolo
dstool voc2yolo -src ./VOCdevkit/Annotations -output ./yolo
```

---

### dstool json2mask

将 LabelMe 格式的 JSON 标注文件转换为像素级二值掩码图像（PNG 格式）。

```bash
dstool json2mask -src ./json_labels -output ./masks
```

**转换逻辑**：

- 每个标注形状绘制为白色（255）填充区域
- 同一类别的不同实例合并到同一张掩码
- 输出为 8-bit 灰度 PNG，背景黑色（0），前景白色（255）

| Shape 类型 | 绘制方式 |
|-----------|---------|
| `rectangle` | 实心矩形填充 |
| `polygon` | 多边形顶点围成的区域填充 |
| `circle` | 圆形/椭圆填充 |

**输出结构**：

```
masks/
├── image001_person.png       # image001 中 person 类别的掩码
├── image001_car.png          # image001 中 car 类别的掩码
├── image002_person.png
├── image002_bicycle.png
├── ...
└── label_colors.txt          # 类别颜色映射参考
```

**掩码文件命名规则**：`{图片名}_{类别名}.png`

---

## LabelMe JSON 输入格式

dstool 接受标准的 **LabelMe** 标注格式。每个 JSON 文件对应一张图片的标注信息。

### 完整 JSON 结构

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[100, 200], [300, 500]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "car",
      "points": [[400, 300], [480, 320], [500, 400], [420, 420]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "logo",
      "points": [[600, 400], [650, 400]],
      "group_id": null,
      "shape_type": "circle",
      "flags": {}
    }
  ],
  "imagePath": "scene_001.jpg",
  "imageData": null,
  "imageHeight": 800,
  "imageWidth": 1200
}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `version` | string | 否 | LabelMe 版本号 |
| `shapes` | array | **是** | 标注对象列表 |
| `shapes[].label` | string | **是** | 类别名称 |
| `shapes[].points` | array | **是** | 坐标点列表 `[[x,y], ...]` |
| `shapes[].shape_type` | string | **是** | 形状类型：`rectangle` / `polygon` / `circle` |
| `shapes[].group_id` | any | 否 | 实例分组 ID |
| `imagePath` | string | **是** | 对应的图片文件名 |
| `imageHeight` | number | **是** | 图片高度（像素） |
| `imageWidth` | number | **是** | 图片宽度（像素） |
| `imageData` | string | 否 | Base64 编码的图片数据（dstool 不处理此字段） |

### shape_type 详解

**rectangle（矩形框）**

由两个对角点定义，通常为左上角和右下角：

```json
{
  "label": "cat",
  "points": [[50, 80], [280, 320]],
  "shape_type": "rectangle"
}
```

**polygon（多边形）**

由 3 个或更多顶点围成的闭合区域：

```json
{
  "label": "building",
  "points": [[100, 200], [300, 180], [320, 400], [80, 420]],
  "shape_type": "polygon"
}
```
- 转 VOC/YOLO 时自动取外接矩形
- 转掩码时保留原始多边形形状

**circle（圆形）**

第一个点为中心点，第二个点确定半径：

```json
{
  "label": "ball",
  "points": [[500, 400], [560, 400]],
  "shape_type": "circle"
}
```
- 半径 = max(|x2-x1|, |y2-y1|)（取 x 和 y 方向偏差的较大值）

---

## 输出格式说明

### Pascal VOC 格式 (json2voc 输出)

```
VOCdevkit/
├── Annotations/
│   ├── image001.xml
│   ├── image002.xml
│   └── ...
├── JPEGImages/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── ImageSets/
    └── Main/
        └── train.txt    # 所有图片的文件名（不含后缀），每行一个
```

VOC XML 文件符合 Pascal VOC 2012 标准，可直接用于：

- Faster R-CNN、SSD、YOLO（需要先转换）等目标检测模型
- 标注工具（如 LabelImg）打开和编辑

### YOLO 格式 (json2yolo / voc2yolo 输出)

```
YOLO_dataset/
├── labels/
│   ├── image001.txt    # class_id cx cy w h（归一化）
│   └── ...
├── classes.txt         # 类别名称
└── dataset.yaml        # 训练配置
```

YOLO txt 标注文件可直接用于 YOLOv5、YOLOv8、YOLOv11 等模型的训练。

### 像素掩码格式 (json2mask 输出)

```
masks/
├── image001_person.png    # 8-bit 灰度 PNG
├── image001_car.png       # 0=背景, 255=前景
├── ...
└── label_colors.txt       # 类别 RGB 颜色参考
```

掩码为单通道 8-bit PNG 图像，可直接用于：

- 语义分割 / 实例分割模型训练（U-Net、Mask R-CNN 等）
- 在图像编辑软件中作为选区叠加查看

---

## 使用场景示例

### 场景 1：从 LabelMe 标注到 YOLO 训练

```bash
# 1. 用 LabelMe 标注完成后，导出 JSON 文件到 labels/ 目录
# 2. 将原始图片放到 images/ 目录

# 3. 转换标注
dstool json2yolo -src ./labels -output ./dataset

# 4. 手动将图片复制到 dataset/images/
cp ./images/* ./dataset/images/

# 5. 编辑 dataset.yaml 中的路径和训练参数
# 然后直接用 YOLOv8 训练：
# yolo train data=./dataset/dataset.yaml model=yolov8n.pt
```

### 场景 2：标注格式链式转换

```bash
# LabelMe JSON → VOC XML → YOLO
dstool json2voc  -src ./labels -output ./VOCdevkit
dstool voc2yolo   -src ./VOCdevkit -output ./yolo_final
```

### 场景 3：生成分割掩码

```bash
# 从多边形标注生成语义分割掩码
dstool json2mask -src ./polygon_labels -output ./seg_masks

# 掩码可用于训练 U-Net 等分割模型
```

### 场景 4：批量处理脚本

```bash
#!/bin/bash
# 批量处理多个数据集文件夹
for dataset in dataset_*; do
    echo "处理 $dataset ..."
    dstool json2yolo -src "./$dataset/json" -output "./${dataset}_yolo"
    dstool json2voc  -src "./$dataset/json" -output "./${dataset}_voc"
done
echo "全部完成！"
```

---

## 依赖

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.8 | 运行环境 |
| Pillow | 9.0.0 | 掩码图像绘制与保存 |
| numpy | 1.20.0 | 数组运算 |
| opencv-python | 4.5.0 | （预留）高级图像处理 |
| tqdm | 4.60.0 | 进度条显示 |

`pip install dstool/` 会自动安装所有 Python 依赖。

---

## 常见问题

### Q: 提示"未找到 JSON/XML 文件"？

检查源路径是否正确，注意 dstool 会**递归搜索**子目录中的所有文件。确保文件后缀名为 `.json` 或 `.xml`。

### Q: 为什么生成的 VOC XML 或 YOLO txt 中某些标注丢失了？

以下情况标注会被跳过：
- 缺少 `label` 字段或 `label` 为空
- `shape_type` 不是 `rectangle`/`polygon`/`circle`
- points 不足（rectangle 需要 ≥2 个点）
- 缺少 `imageHeight` 或 `imageWidth` 字段

### Q: 掩码图像中同一类别有多个实例，它们会分开吗？

默认情况下，同一图片中**同一类别**的所有实例会合并到**同一张**掩码中。如果需要实例级掩码（每个实例独立），目前请确保标注时不同实例使用不同的 `label`（如 `person_1`, `person_2`）。

### Q: json2voc 会复制图片吗？

会尝试复制，但依赖 JSON 中的 `imagePath` 字段能找到对应的图片文件。dstool 会在以下位置查找：

1. JSON 文件所在目录
2. 源目录根目录
3. JSON 中的 `imagePath` 的原始路径

如果图片不存在，不会影响 XML 的生成，只是 `JPEGImages/` 目录下没有图片。

### Q: Windows 终端输出乱码？

dstool 已内置编码兼容处理。如仍有乱码，请在终端中执行 `chcp 65001` 切换到 UTF-8 编码。

---

## License

MIT License

---

## 项目结构

```
dstool/
├── pyproject.toml
├── README.md
├── dstool/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # CLI 主入口
│   ├── utils.py             # 公共工具函数
│   └── converters/
│       ├── __init__.py
│       ├── json2voc.py      # LabelMe JSON → VOC XML
│       ├── json2yolo.py     # LabelMe JSON → YOLO txt
│       ├── voc2yolo.py      # VOC XML → YOLO txt
│       └── json2mask.py     # LabelMe JSON → Pixel Mask PNG
└── test_data/
    └── json_labels/         # 测试用 LabelMe JSON 示例
```
