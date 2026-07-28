"""dstool 公共工具函数"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


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

    采用多级搜索策略，尽可能找到对应图片。

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

    json_dir = os.path.dirname(json_path)
    base = os.path.basename(image_name)

    # 阶段1：精确候选路径匹配
    candidates = [
        # JSON 同目录
        os.path.join(json_dir, image_name),
        os.path.join(json_dir, base),
        # JSON 同目录 + imagePath 作为相对路径
        os.path.normpath(os.path.join(json_dir, image_name)),
        # 源目录根
        os.path.join(source_dir, image_name),
        os.path.join(source_dir, base),
        # 源目录下的 images 子目录
        os.path.join(source_dir, "images", image_name),
        os.path.join(source_dir, "images", base),
        # JSON 同目录下的 images 子目录
        os.path.join(json_dir, "images", image_name),
        os.path.join(json_dir, "images", base),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # 阶段2：在源目录下递归搜索（按文件名匹配）
    # 支持常见图片扩展名
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    base_no_ext = os.path.splitext(base)[0].lower()

    for root, _, files in os.walk(source_dir):
        for f in files:
            # 精确匹配文件名
            if f == base:
                return os.path.join(root, f)
            # 忽略扩展名大小写匹配
            f_lower = f.lower()
            name_lower = base.lower()
            if f_lower == name_lower:
                return os.path.join(root, f)
            # 同名但不同扩展名
            f_no_ext = os.path.splitext(f)[0].lower()
            f_ext = os.path.splitext(f)[1].lower()
            if f_no_ext == base_no_ext and f_ext in img_extensions:
                return os.path.join(root, f)

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


def get_class_colors(classes: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """为每个类别分配唯一颜色（基于 HSV 色彩空间均匀分布）

    Args:
        classes: 类别名称列表

    Returns:
        {class_name: (R, G, B)} 颜色映射字典
    """
    colors = {}
    n = max(len(classes), 1)
    for i, name in enumerate(classes):
        hue = i / n
        r, g, b = _hsv_to_rgb(hue, 1.0, 1.0)
        colors[name] = (r, g, b)
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """HSV 转 RGB (0-255)

    Args:
        h: 色调 (0-1)
        s: 饱和度 (0-1)
        v: 明度 (0-1)

    Returns:
        (r, g, b) 各通道 0-255
    """
    if s == 0:
        val = int(v * 255)
        return val, val, val

    h = h * 6.0
    i = int(h)
    f = h - i
    p = int(v * (1.0 - s) * 255)
    q = int(v * (1.0 - s * f) * 255)
    t = int(v * (1.0 - s * (1.0 - f)) * 255)
    val = int(v * 255)

    if i == 0:
        return val, t, p
    elif i == 1:
        return q, val, p
    elif i == 2:
        return p, val, t
    elif i == 3:
        return p, q, val
    elif i == 4:
        return t, p, val
    else:
        return val, p, q


def generate_bbox_visualization(image_path: str, bboxes: List[Dict],
                                  class_colors: Dict[str, Tuple[int, int, int]],
                                  output_path: str) -> bool:
    """在原图上绘制 bbox 标注框和标签，生成可视化图片

    Args:
        image_path: 原图路径
        bboxes: bbox 列表，每项含 label, xmin, ymin, xmax, ymax
        class_colors: {class_name: (R, G, B)} 颜色映射
        output_path: 输出图片路径

    Returns:
        成功返回 True，失败返回 False
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(img)

    # 尝试加载字体，失败则使用默认字体
    font = None
    for font_name in ["arial.ttf", "simhei.ttf", "msyh.ttc", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 14)
            break
        except Exception:
            continue

    for bbox in bboxes:
        label = bbox["label"]
        xmin, ymin = int(bbox["xmin"]), int(bbox["ymin"])
        xmax, ymax = int(bbox["xmax"]), int(bbox["ymax"])

        color = class_colors.get(label, (255, 0, 0))

        # 画矩形框 (2px 宽)
        for offset in range(2):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=color
            )

        # 画标签背景色块 + 文字
        text = label
        if font:
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
            except Exception:
                text_bbox = (0, 0, len(text) * 8, 14)
        else:
            text_bbox = (0, 0, len(text) * 8, 14)

        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, ymin - text_h - 4)

        draw.rectangle(
            [xmin, label_y, xmin + text_w + 4, label_y + text_h + 4],
            fill=color
        )
        draw.text((xmin + 2, label_y + 2), text, fill=(255, 255, 255), font=font)

    try:
        img.save(output_path)
        return True
    except Exception:
        return False


def generate_mask_visualization(image_path: str,
                                  masks: Dict[str, "np.ndarray"],
                                  class_colors: Dict[str, Tuple[int, int, int]],
                                  output_path: str,
                                  alpha: float = 0.4) -> bool:
    """在原图上叠加彩色掩码，生成可视化图片

    Args:
        image_path: 原图路径
        masks: {class_name: numpy_mask_array} 掩码字典，mask 值为 0 或 255
        class_colors: {class_name: (R, G, B)} 颜色映射
        output_path: 输出图片路径
        alpha: 掩码叠加透明度 (0~1)

    Returns:
        成功返回 True，失败返回 False
    """
    import numpy as np

    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception:
        return False

    img_array = np.array(img, dtype=np.float64)

    for label, mask_array in masks.items():
        if mask_array is None or mask_array.size == 0:
            continue
        color = class_colors.get(label, (255, 0, 0))
        fg = mask_array > 0

        for c in range(3):  # R, G, B 三个通道
            channel = img_array[:, :, c].copy()
            channel[fg] = channel[fg] * (1 - alpha) + color[c] * alpha
            img_array[:, :, c] = channel

    result = Image.fromarray(img_array.astype(np.uint8), mode="RGBA")
    # 转回 RGB 保存
    try:
        result.convert("RGB").save(output_path)
        return True
    except Exception:
        return False
