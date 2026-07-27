"""LabelMe JSON → YOLO 格式转换器"""

import os
from typing import Dict, Any

from tqdm import tqdm

from dstool.utils import (
    find_json_files,
    load_labelme_json,
    shapes_to_bboxes,
    collect_class_names,
    yolo_bbox,
    make_output_dir,
)


def convert_json2yolo(source_dir: str, output_dir: str) -> Dict[str, Any]:
    """将 LabelMe JSON 标注转换为 YOLO 格式

    YOLO 输出结构:
        output_dir/
        ├── images/
        ├── labels/
        ├── classes.txt
        └── dataset.yaml

    Args:
        source_dir: 包含 JSON 文件的源目录
        output_dir: 输出目录

    Returns:
        包含转换统计信息的字典
    """
    source_dir = os.path.abspath(source_dir)
    output_dir = make_output_dir(output_dir)

    labels_dir = make_output_dir(os.path.join(output_dir, "labels"))

    json_files = find_json_files(source_dir)
    if not json_files:
        print(f"错误: 在 {source_dir} 中未找到 JSON 文件")
        return {"total": 0, "objects": 0, "classes": []}

    print(f"找到 {len(json_files)} 个 JSON 文件")

    all_data = []
    for jf in json_files:
        data = load_labelme_json(jf)
        if data is not None:
            all_data.append((jf, data))

    if not all_data:
        print("错误: 没有有效的 JSON 标注文件")
        return {"total": 0, "objects": 0, "classes": []}

    classes = collect_class_names(all_data)
    class_to_id = {name: i for i, name in enumerate(classes)}

    total_objects = 0
    processed = 0

    for json_path, data in tqdm(all_data, desc="转换 JSON→YOLO"):
        image_filename = os.path.basename(data.get("imagePath", ""))
        if not image_filename:
            image_filename = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"

        img_width = int(data.get("imageWidth", 0))
        img_height = int(data.get("imageHeight", 0))

        if img_width <= 0 or img_height <= 0:
            print(f"  警告: {json_path} 图像尺寸无效，跳过")
            continue

        bboxes = shapes_to_bboxes(data.get("shapes", []))
        if not bboxes:
            continue

        # 生成 YOLO 格式标注文件
        label_name = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w", encoding="utf-8") as f:
            for bbox in bboxes:
                cx, cy, w, h = yolo_bbox(
                    bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"],
                    img_width, img_height
                )
                class_id = class_to_id[bbox["label"]]
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        total_objects += len(bboxes)
        processed += 1

    # 生成 classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in classes:
            f.write(f"{name}\n")

    # 生成 dataset.yaml (YOLOv5/v8 风格)
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# YOLO 数据集配置文件\n")
        f.write(f"# 由 dstool 自动生成\n\n")
        f.write(f"path: {os.path.abspath(output_dir).replace(os.sep, '/')}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    return {
        "total": processed,
        "objects": total_objects,
        "classes": classes,
    }
