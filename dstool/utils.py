"""dstool 公共工具函数"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def find_json_files(source_dir: str) -> List[str]:
    """递归查找目录下所有 .json 文件（LabelMe 格式）

    Args:
        source_dir: 源目录路径

    Returns:
        JSON 文件路径列表
    """
    json_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))
    return json_files


def find_xml_files(source_dir: str, use_voc_structure: bool = False) -> List[str]:
    """查找 XML 文件

    Args:
        source_dir: 源目录路径
        use_voc_structure: 是否按 VOC 目录结构查找 (Annotations/ 子目录)

    Returns:
        XML 文件路径列表
    """
    xml_files = []

    if use_voc_structure:
        # 先尝试 VOC 标准结构: Annotations/ 目录
        ann_dir = os.path.join(source_dir, "Annotations")
        if os.path.isdir(ann_dir):
            for root, _, files in os.walk(ann_dir):
                for f in files:
                    if f.lower().endswith(".xml"):
                        xml_files.append(os.path.join(root, f))
            if xml_files:
                return xml_files

    # 递归搜索所有 .xml 文件
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    return xml_files


def load_labelme_json(json_path: str) -> Optional[Dict]:
    """加载并验证 LabelMe 格式 JSON 文件

    Args:
        json_path: JSON 文件路径

    Returns:
        解析后的字典，解析失败返回 None
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 验证基本结构
        if "shapes" not in data:
            print(f"  警告: {json_path} 缺少 'shapes' 字段，跳过")
            return None
        if "imageHeight" not in data or "imageWidth" not in data:
            print(f"  警告: {json_path} 缺少图像尺寸信息，跳过")
            return None
        return data
    except Exception as e:
        print(f"  错误: 无法解析 {json_path}: {e}")
        return None


def get_image_path_from_json(json_path: str, json_data: Dict, source_dir: str) -> Optional[str]:
    """根据 JSON 中的 imagePath 查找对应的图片文件

    Args:
        json_path: JSON 文件的路径
        json_data: 解析后的 JSON 数据
        source_dir: 源目录

    Returns:
        图片文件路径，找不到返回 None
    """
    image_name = json_data.get("imagePath", "")
    if not image_name:
        return None

    # 尝试多种路径组合
    candidates = [
        os.path.join(os.path.dirname(json_path), image_name),
        os.path.join(os.path.dirname(json_path), os.path.basename(image_name)),
        os.path.join(source_dir, image_name),
        os.path.join(source_dir, os.path.basename(image_name)),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def collect_class_names(all_data: List[Tuple[str, Dict]]) -> List[str]:
    """从所有 JSON 数据中收集类别名称，按字母排序

    Args:
        all_data: [(json_path, json_data), ...] 列表

    Returns:
        排序后的类别名称列表
    """
    classes = set()
    for _, data in all_data:
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label:
                classes.add(label)
    return sorted(classes)


def shapes_to_bboxes(shapes: List[Dict]) -> List[Dict]:
    """将 LabelMe shapes 转换为统一的 bbox 格式

    支持 rectangle、polygon、circle 三种 shape_type。
    返回的 bbox 格式: {"label": str, "xmin": int, "ymin": int, "xmax": int, "ymax": int}

    Args:
        shapes: LabelMe shapes 列表

    Returns:
        bbox 列表
    """
    bboxes = []
    for shape in shapes:
        label = shape.get("label", "").strip()
        if not label:
            continue

        shape_type = shape.get("shape_type", "rectangle")
        points = shape.get("points", [])

        if len(points) < 2 and shape_type != "circle":
            continue

        xmin, ymin, xmax, ymax = _compute_bbox(points, shape_type)

        if xmax > xmin and ymax > ymin:
            bboxes.append({
                "label": label,
                "xmin": int(round(xmin)),
                "ymin": int(round(ymin)),
                "xmax": int(round(xmax)),
                "ymax": int(round(ymax)),
            })

    return bboxes


def _compute_bbox(points: List[List[float]], shape_type: str) -> Tuple[float, float, float, float]:
    """根据 shape_type 计算边界框

    Args:
        points: 坐标点列表
        shape_type: shape 类型 (rectangle, polygon, circle)

    Returns:
        (xmin, ymin, xmax, ymax)
    """
    if shape_type == "rectangle":
        # rectangle: 两个对角点
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    elif shape_type == "polygon":
        # polygon: 多个顶点
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    elif shape_type == "circle":
        # circle: 第一个点是圆心，从第一个点到第二个点的距离为半径
        # 实际上 LabelMe 的 circle 通常只存储中心点和边缘一点
        cx, cy = points[0]
        if len(points) >= 2:
            rx = abs(points[1][0] - cx)
            ry = abs(points[1][1] - cy)
            r = max(rx, ry)
        else:
            r = 10  # 默认半径
        return cx - r, cy - r, cx + r, cy + r

    else:
        # 默认为 polygon 处理
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)


def yolo_bbox(xmin: int, ymin: int, xmax: int, ymax: int,
              img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """将像素坐标 bbox 转换为 YOLO 归一化格式

    Args:
        xmin, ymin, xmax, ymax: 像素坐标
        img_width, img_height: 图像尺寸

    Returns:
        (cx, cy, w, h) 归一化到 0~1
    """
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    cx = (xmin + xmax) / 2.0 / img_width
    cy = (ymin + ymax) / 2.0 / img_height

    # 限制在 [0, 1] 范围内
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def copy_images(src: str, dst: str, image_paths: List[str]) -> int:
    """复制图片文件到目标目录

    Args:
        src: 源目录 (未使用，保留参数)
        dst: 目标目录
        image_paths: 图片路径列表

    Returns:
        成功复制的文件数
    """
    os.makedirs(dst, exist_ok=True)
    count = 0
    for img_path in image_paths:
        if img_path and os.path.isfile(img_path):
            dst_path = os.path.join(dst, os.path.basename(img_path))
            if not os.path.exists(dst_path):
                shutil.copy2(img_path, dst_path)
                count += 1
    return count


def make_output_dir(path: str) -> str:
    """创建输出目录并返回绝对路径

    Args:
        path: 目录路径

    Returns:
        目录的绝对路径
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)
