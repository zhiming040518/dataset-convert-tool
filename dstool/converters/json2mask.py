"""LabelMe JSON → 像素掩码 转换器"""

import os
from typing import Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from dstool.utils import (
    find_json_files,
    load_labelme_json,
    get_image_path_from_json,
    collect_class_names,
    copy_images,
    extract_image_from_base64,
    get_class_colors,
    generate_mask_visualization,
    make_output_dir,
)


def _draw_mask(shapes: list, img_width: int, img_height: int) -> Dict[str, np.ndarray]:
    """为每个类别绘制合并的二值掩码

    Args:
        shapes: LabelMe shapes 列表
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        {label: mask_array} 每个类别的二值掩码 (0/255, uint8)
    """
    from PIL import ImageDraw

    # 按类别分组
    label_masks = {}

    for shape in shapes:
        label = shape.get("label", "").strip()
        if not label:
            continue

        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        if len(points) < 2:
            continue

        # 创建单张掩码
        mask = Image.new("L", (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)

        if shape_type == "rectangle":
            # rectangle: 两个对角点 → 填充矩形
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            # Pillow rectangle 需要 (x0, y0, x1, y1)
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        elif shape_type == "circle":
            # circle: 中心点 + 边缘点
            cx, cy = points[0]
            if len(points) >= 2:
                rx = abs(points[1][0] - cx)
                ry = abs(points[1][1] - cy)
                r = max(rx, ry)
            else:
                r = 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        else:
            # polygon 或默认
            # Pillow polygon 需要展平的点列表 [(x1,y1), (x2,y2), ...]
            polygon_pts = [(float(p[0]), float(p[1])) for p in points]
            draw.polygon(polygon_pts, fill=255)

        # 合并到类别掩码
        if label not in label_masks:
            label_masks[label] = np.zeros((img_height, img_width), dtype=np.uint8)

        mask_array = np.array(mask, dtype=np.uint8)
        label_masks[label] = np.maximum(label_masks[label], mask_array)

    return label_masks


def convert_json2mask(source_dir: str, output_dir: str) -> Dict[str, Any]:
    """将 LabelMe JSON 标注转换为像素掩码图像

    为每个 JSON 文件中每个类别的标注生成对应的二值掩码 PNG 文件。
    掩码文件命名: {image_name}_{label}.png

    输出结构:
        output_dir/
        ├── images/              # 原始图片
        ├── visualizations/      # 掩码叠加可视化图
        ├── {name}_{label}.png   # 各掩码文件
        └── label_colors.txt     # 类别颜色参考

    Args:
        source_dir: 包含 JSON 文件的源目录
        output_dir: 输出目录

    Returns:
        包含转换统计信息的字典
    """
    source_dir = os.path.abspath(source_dir)
    output_dir = make_output_dir(output_dir)

    img_dir = make_output_dir(os.path.join(output_dir, "images"))
    viz_dir = make_output_dir(os.path.join(output_dir, "visualizations"))

    json_files = find_json_files(source_dir)
    if not json_files:
        print(f"错误: 在 {source_dir} 中未找到 JSON 文件")
        return {"total": 0, "masks": 0, "classes": [], "copied_images": 0, "visualizations": 0}

    print(f"找到 {len(json_files)} 个 JSON 文件")

    all_data = []
    for jf in json_files:
        data = load_labelme_json(jf)
        if data is not None:
            all_data.append((jf, data))

    if not all_data:
        print("错误: 没有有效的 JSON 标注文件")
        return {"total": 0, "masks": 0, "classes": [], "copied_images": 0, "visualizations": 0}

    classes = collect_class_names(all_data)
    class_colors = get_class_colors(classes)
    total_masks = 0
    processed = 0
    image_paths = []
    viz_count = 0

    for json_path, data in tqdm(all_data, desc="转换 JSON→Mask"):
        image_filename = os.path.basename(data.get("imagePath", ""))
        if not image_filename:
            image_filename = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"

        base_name = os.path.splitext(image_filename)[0]
        img_width = int(data.get("imageWidth", 0))
        img_height = int(data.get("imageHeight", 0))

        if img_width <= 0 or img_height <= 0:
            print(f"  警告: {json_path} 图像尺寸无效，跳过")
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        label_masks = _draw_mask(shapes, img_width, img_height)

        for label, mask_array in label_masks.items():
            # 安全的文件名
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            mask_filename = f"{base_name}_{safe_label}.png"
            mask_path = os.path.join(output_dir, mask_filename)

            mask_img = Image.fromarray(mask_array, mode="L")
            mask_img.save(mask_path)

            total_masks += 1

        processed += 1

        # 查找图片路径
        img_path = get_image_path_from_json(json_path, data, source_dir)
        if not img_path:
            # 兜底：尝试从 base64 imageData 提取
            img_filename = os.path.basename(image_filename)
            base64_out = os.path.join(img_dir, img_filename)
            if extract_image_from_base64(data, base64_out):
                img_path = base64_out
        image_paths.append(img_path)

        # 生成掩码叠加可视化
        if img_path and os.path.isfile(img_path) and label_masks:
            viz_path = os.path.join(viz_dir, f"{base_name}.jpg")
            if generate_mask_visualization(img_path, label_masks, class_colors, viz_path):
                viz_count += 1

    # 复制图片文件
    copied = copy_images(source_dir, img_dir, image_paths)
    if copied:
        print(f"  复制了 {copied} 张图片到 {img_dir}")
    if viz_count:
        print(f"  生成了 {viz_count} 张可视化图到 {viz_dir}")

    # 保存类别颜色映射文件
    _save_label_colors(output_dir, classes)

    return {
        "total": processed,
        "masks": total_masks,
        "classes": classes,
        "copied_images": copied,
        "visualizations": viz_count,
    }


def _save_label_colors(output_dir: str, classes: list):
    """为每个类别生成不同颜色的映射文件

    Args:
        output_dir: 输出目录
        classes: 类别列表
    """
    if not classes:
        return

    colors_path = os.path.join(output_dir, "label_colors.txt")
    with open(colors_path, "w", encoding="utf-8") as f:
        f.write("# 类别颜色映射 (RGB)\n")
        f.write("# 格式: class_name R G B\n\n")

        for i, name in enumerate(classes):
            # 使用 HSV 色彩空间均匀分配颜色
            hue = i / max(len(classes), 1)
            # HSV → RGB 简化转换
            r, g, b = _hsv_to_rgb(hue, 1.0, 1.0)
            f.write(f"{name} {r} {g} {b}\n")


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
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
