"""LabelMe JSON → Pascal VOC XML 转换器"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any

from tqdm import tqdm

from dstool.utils import (
    find_json_files,
    load_labelme_json,
    get_image_path_from_json,
    shapes_to_bboxes,
    collect_class_names,
    copy_images,
    extract_image_from_base64,
    get_class_colors,
    generate_bbox_visualization,
    make_output_dir,
)


def _create_voc_xml(json_data: Dict, image_filename: str, bboxes: list) -> str:
    """创建 Pascal VOC 格式的 XML 字符串

    Args:
        json_data: LabelMe JSON 数据
        image_filename: 图片文件名
        bboxes: bbox 列表

    Returns:
        格式化的 XML 字符串
    """
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2007"

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename

    # 图像尺寸
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(int(json_data.get("imageWidth", 0)))
    height = ET.SubElement(size, "height")
    height.text = str(int(json_data.get("imageHeight", 0)))
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    # 每个标注对象
    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = bbox["label"]
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox["xmin"]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox["ymin"]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox["xmax"]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox["ymax"]))

    # 格式化输出
    rough = ET.tostring(annotation, "utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def convert_json2voc(source_dir: str, output_dir: str) -> Dict[str, Any]:
    """将 LabelMe JSON 标注转换为 Pascal VOC XML 格式

    Args:
        source_dir: 包含 JSON 文件的源目录
        output_dir: 输出目录

    Returns:
        包含转换统计信息的字典
    """
    source_dir = os.path.abspath(source_dir)
    output_dir = make_output_dir(output_dir)

    ann_dir = make_output_dir(os.path.join(output_dir, "Annotations"))
    img_dir = make_output_dir(os.path.join(output_dir, "JPEGImages"))
    viz_dir = make_output_dir(os.path.join(output_dir, "visualizations"))
    sets_dir = make_output_dir(os.path.join(output_dir, "ImageSets", "Main"))

    json_files = find_json_files(source_dir)
    if not json_files:
        print(f"错误: 在 {source_dir} 中未找到 JSON 文件")
        return {"total": 0, "objects": 0, "classes": [], "copied_images": 0, "visualizations": 0}

    print(f"找到 {len(json_files)} 个 JSON 文件")

    all_data = []
    for jf in json_files:
        data = load_labelme_json(jf)
        if data is not None:
            all_data.append((jf, data))

    if not all_data:
        print("错误: 没有有效的 JSON 标注文件")
        return {"total": 0, "objects": 0, "classes": [], "copied_images": 0, "visualizations": 0}

    classes = collect_class_names(all_data)
    class_colors = get_class_colors(classes)
    total_objects = 0
    processed = 0
    image_paths = []
    viz_count = 0

    for json_path, data in tqdm(all_data, desc="转换 JSON→VOC"):
        image_filename = os.path.basename(data.get("imagePath", ""))
        if not image_filename:
            image_filename = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"

        bboxes = shapes_to_bboxes(data.get("shapes", []))
        if not bboxes:
            continue

        # 生成 XML
        xml_str = _create_voc_xml(data, image_filename, bboxes)
        xml_filename = os.path.splitext(image_filename)[0] + ".xml"
        xml_path = os.path.join(ann_dir, xml_filename)

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        total_objects += len(bboxes)
        processed += 1

        # 记录图片路径
        img_path = get_image_path_from_json(json_path, data, source_dir)
        if not img_path:
            # 兜底：尝试从 base64 imageData 提取
            img_filename = os.path.basename(image_filename)
            base64_out = os.path.join(img_dir, img_filename)
            if extract_image_from_base64(data, base64_out):
                img_path = base64_out
        image_paths.append(img_path)

        # 生成可视化图片
        if img_path and os.path.isfile(img_path):
            img_base = os.path.splitext(image_filename)[0]
            viz_path = os.path.join(viz_dir, f"{img_base}.jpg")
            if generate_bbox_visualization(img_path, bboxes, class_colors, viz_path):
                viz_count += 1

    # 复制图片文件
    copied = copy_images(source_dir, img_dir, image_paths)
    if copied:
        print(f"  复制了 {copied} 张图片到 {img_dir}")

    if viz_count:
        print(f"  生成了 {viz_count} 张可视化图到 {viz_dir}")

    # 生成 ImageSets/Main/train.txt
    img_names = []
    for json_path, data in all_data:
        img_name = os.path.basename(data.get("imagePath", ""))
        if not img_name:
            img_name = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
        img_names.append(os.path.splitext(img_name)[0])

    if img_names:
        with open(os.path.join(sets_dir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(img_names) + "\n")

    return {
        "total": processed,
        "objects": total_objects,
        "classes": classes,
        "copied_images": copied,
        "visualizations": viz_count,
    }
