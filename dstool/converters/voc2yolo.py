"""VOC XML → YOLO 格式转换器"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, Any

from tqdm import tqdm

from dstool.utils import (
    find_xml_files,
    yolo_bbox,
    make_output_dir,
)


def _parse_voc_xml(xml_path: str) -> dict:
    """解析 Pascal VOC XML 文件

    Args:
        xml_path: XML 文件路径

    Returns:
        包含 filename, width, height, objects 的字典
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename")
        filename = filename.text if filename is not None else ""

        size = root.find("size")
        width = int(size.find("width").text) if size is not None and size.find("width") is not None else 0
        height = int(size.find("height").text) if size is not None and size.find("height") is not None else 0

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name")
            name = name.text if name is not None else ""

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find("xmin").text)) if bndbox.find("xmin") is not None else 0
            ymin = int(float(bndbox.find("ymin").text)) if bndbox.find("ymin") is not None else 0
            xmax = int(float(bndbox.find("xmax").text)) if bndbox.find("xmax") is not None else 0
            ymax = int(float(bndbox.find("ymax").text)) if bndbox.find("ymax") is not None else 0

            objects.append({
                "label": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        }
    except Exception as e:
        print(f"  警告: 解析 {xml_path} 失败: {e}")
        return None


def convert_voc2yolo(source_dir: str, output_dir: str) -> Dict[str, Any]:
    """将 VOC XML 标注转换为 YOLO 格式

    YOLO 输出结构:
        output_dir/
        ├── labels/
        ├── classes.txt
        └── dataset.yaml

    Args:
        source_dir: 包含 XML 文件的源目录（支持自动检测 VOC 目录结构）
        output_dir: 输出目录

    Returns:
        包含转换统计信息的字典
    """
    source_dir = os.path.abspath(source_dir)
    output_dir = make_output_dir(output_dir)

    labels_dir = make_output_dir(os.path.join(output_dir, "labels"))

    # 自动检测 VOC 目录结构
    xml_files = find_xml_files(source_dir, use_voc_structure=True)
    if not xml_files:
        print(f"错误: 在 {source_dir} 中未找到 XML 文件")
        print(f"提示: 如果是VOC格式，确保 XML 文件在 Annotations/ 目录下")
        return {"total": 0, "objects": 0, "classes": []}

    print(f"找到 {len(xml_files)} 个 XML 文件")

    # 收集所有类别
    class_set = set()
    all_parsed = []

    for xf in xml_files:
        parsed = _parse_voc_xml(xf)
        if parsed is not None and parsed["objects"]:
            all_parsed.append(parsed)
            for obj in parsed["objects"]:
                if obj["label"]:
                    class_set.add(obj["label"])

    if not all_parsed:
        print("错误: 没有有效的 XML 标注文件")
        return {"total": 0, "objects": 0, "classes": []}

    classes = sorted(class_set)
    class_to_id = {name: i for i, name in enumerate(classes)}

    total_objects = 0
    processed = 0

    for parsed in tqdm(all_parsed, desc="转换 VOC→YOLO"):
        filename = parsed["filename"]
        img_width = parsed["width"]
        img_height = parsed["height"]

        if img_width <= 0 or img_height <= 0:
            print(f"  警告: {filename} 图像尺寸无效，跳过")
            continue

        # 生成 YOLO 标注文件
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w", encoding="utf-8") as f:
            for obj in parsed["objects"]:
                cx, cy, w, h = yolo_bbox(
                    obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                    img_width, img_height
                )
                class_id = class_to_id[obj["label"]]
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        total_objects += len(parsed["objects"])
        processed += 1

    # 生成 classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in classes:
            f.write(f"{name}\n")

    # 生成 dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# YOLO 数据集配置文件\n")
        f.write(f"# 由 dstool 自动生成 (从 VOC 转换)\n\n")
        f.write(f"path: {os.path.abspath(output_dir).replace(os.sep, '/')}\n")
        f.write(f"train: labels\n")
        f.write(f"val: labels\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    return {
        "total": processed,
        "objects": total_objects,
        "classes": classes,
    }
