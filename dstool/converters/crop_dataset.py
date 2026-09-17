"""按目标裁剪数据集（YOLO / VOC），每个目标生成一张固定尺寸的小图

裁剪窗口按「相对位置等比滑动」定位：设目标框中心在原图中的相对位置为 (rx, ry)，
则窗口左上角 ox = rx × (原图宽 − 窗口边长)，oy 同理。于是:

    - 目标中心在裁剪图中的相对位置仍然是 (rx, ry)，与原图完全一致；
    - 目标不会被裁出窗口（目标靠左时窗口贴左边缘，靠右时贴右边缘）。

一张原图有 N 个目标就裁出 N 张图（数据集会变大），因此只有「输出到新目录」一种模式。
每张裁剪图的标签里保留窗口内所有可见目标，被窗口切到的框钳制成可见部分。
"""

import os
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageOps

from dstool.converters.json2voc import _create_voc_xml
from dstool.converters.merge_voc import _resolve_voc_dirs
from dstool.converters.merge_yolo import (
    _get_dataset_classes,
    _list_image_files,
)
from dstool.converters.rename_yolo import (
    _discover_splits,
    _has_ext,
    _looks_like_dir,
    _new_stem,
    sanitize_prefix,
)
from dstool.converters.voc2yolo import _parse_voc_xml
from dstool.utils import (
    draw_bboxes,
    generate_bbox_visualization_from_image,
    get_class_colors,
    load_visualization_font,
    make_output_dir,
    yolo_bbox,
)

# 钳制后可见宽或高小于该值（像素）的框直接丢弃
MIN_VISIBLE = 1.0

# 默认裁剪尺寸与 JPEG 质量
DEFAULT_SIZE = 512
DEFAULT_QUALITY = 95

# 对比图左面板（原图）的最大宽度，避免超宽原图产生巨型画布
MAX_PANEL_W = 1024

# EXIF 方向标签
EXIF_ORIENTATION = 0x0112
# 需要交换宽高的 EXIF 方向值
ROTATED_ORIENTATIONS = (5, 6, 7, 8)

# 支持的输出图片格式
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class SourceBox(NamedTuple):
    """原图坐标系下的标注框（浮点像素）"""
    label: str
    class_id: Optional[int]  # 仅 YOLO 有，VOC 为 None
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class CropItem(NamedTuple):
    """一张待生成的裁剪图"""
    src_image: str
    src_index: int                 # 参考目标在源框列表中的下标
    ox: int                        # 窗口左上角
    oy: int
    boxes: List[Dict[str, Any]]    # 该裁剪图自己的标注（已变换到窗口坐标系）
    anchor_line: int               # 参考目标在输出标签中的行号
    n_dropped: int                 # 本窗口丢弃的框数
    n_clipped: int                 # 本窗口被切边的框数


class Split:
    """一个待处理的分集"""
    def __init__(self, token: Optional[str], images_dir: str, labels_dir: str,
                 stems: Optional[Set[str]] = None):
        self.token = token
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.stems = stems  # 仅 VOC 的 ImageSets 会限制处理范围


# ---------------------------------------------------------------- 纯函数


def _window_offset(anchor: SourceBox, width: int, height: int,
                   size: int) -> Tuple[int, int]:
    """计算裁剪窗口左上角：按相对位置等比滑动，并在必要时光滑让开

    理想位置由「目标中心的相对位置在裁剪图中保持不变」推出：
        ox = rx × (原图宽 − 窗口边长)
    若该位置会把目标本身切出窗口（目标贴着原图边缘时必然发生），
    则把窗口在允许范围内滑动到「能让目标完整进框」且最接近理想位置处。

    Args:
        anchor: 参考目标框（原图像素坐标，已钳制到图片范围内）
        width, height: 原图尺寸（必须均 >= size）
        size: 窗口边长

    Returns:
        (ox, oy) 窗口左上角像素坐标
    """
    cx = (anchor.xmin + anchor.xmax) / 2.0
    cy = (anchor.ymin + anchor.ymax) / 2.0

    ox = int(round(cx / width * (width - size)))
    oy = int(round(cy / height * (height - size)))

    ox = _keep_inside(ox, anchor.xmin, anchor.xmax, width, size)
    oy = _keep_inside(oy, anchor.ymin, anchor.ymax, height, size)
    return ox, oy


def _keep_inside(pos: int, low: float, high: float, total: int, size: int) -> int:
    """把窗口位置调整到能让 [low, high] 完整落在窗口内的最接近 pos 处

    需要同时满足 pos <= low 与 pos + size >= high，即 pos ∈ [high - size, low]。
    """
    if high - low >= size:
        # 目标比窗口还大，怎么放都会被切 → 让窗口对准目标中心，切得最均匀
        pos = (low + high) / 2.0 - size / 2.0
    else:
        pos = min(max(float(pos), high - size), float(low))

    return max(0, min(total - size, int(round(pos))))


def _clamp_boxes_to_image(boxes: List[SourceBox],
                          width: int, height: int) -> Tuple[List[SourceBox], int]:
    """把标注框钳制到图片范围内，剔除钳制后已退化的框

    Returns:
        (有效框列表, 无效框数)
    """
    kept: List[SourceBox] = []
    invalid = 0
    for box in boxes:
        xmin = min(max(box.xmin, 0.0), float(width))
        ymin = min(max(box.ymin, 0.0), float(height))
        xmax = min(max(box.xmax, 0.0), float(width))
        ymax = min(max(box.ymax, 0.0), float(height))

        if xmax - xmin < MIN_VISIBLE or ymax - ymin < MIN_VISIBLE:
            invalid += 1
            continue

        kept.append(SourceBox(box.label, box.class_id, xmin, ymin, xmax, ymax))
    return kept, invalid


def _transform_boxes(boxes: List[SourceBox], ox: int, oy: int,
                     size: int) -> Tuple[List[Dict[str, Any]], int, int]:
    """把标注框变换到窗口坐标系：平移 → 与窗口求交 → 取整

    Returns:
        (框列表, 完全落在窗口外被丢弃的数量, 被窗口切边的数量)
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    clipped = 0

    for index, box in enumerate(boxes):
        xmin = box.xmin - ox
        ymin = box.ymin - oy
        xmax = box.xmax - ox
        ymax = box.ymax - oy

        # 先判断是否被窗口边界切到（只碰到边界不算）
        is_clipped = (xmin < 0.0 or ymin < 0.0
                      or xmax > float(size) or ymax > float(size))

        # 与窗口求交，保住可见部分
        vx1 = max(xmin, 0.0)
        vy1 = max(ymin, 0.0)
        vx2 = min(xmax, float(size))
        vy2 = min(ymax, float(size))

        if vx2 - vx1 < MIN_VISIBLE or vy2 - vy1 < MIN_VISIBLE:
            dropped += 1
            continue

        ix1, iy1 = int(round(vx1)), int(round(vy1))
        ix2, iy2 = int(round(vx2)), int(round(vy2))
        if ix2 <= ix1 or iy2 <= iy1:
            dropped += 1
            continue

        if is_clipped:
            clipped += 1

        kept.append({
            "label": box.label,
            "class_id": box.class_id,
            "xmin": ix1,
            "ymin": iy1,
            "xmax": ix2,
            "ymax": iy2,
            "src_index": index,
        })

    return kept, dropped, clipped


# ---------------------------------------------------------------- 读取


def _detect_format(src_dir: str) -> str:
    """识别数据集格式，返回 "voc" 或 "yolo" """
    if _looks_like_dir(os.path.join(src_dir, "Annotations"), {".xml"}):
        return "voc"
    if _has_ext(src_dir, {".xml"}):
        return "voc"
    if os.path.isdir(src_dir):
        for name in sorted(os.listdir(src_dir)):
            sub_dir = os.path.join(src_dir, name)
            if os.path.isdir(sub_dir) and _looks_like_dir(
                    os.path.join(sub_dir, "Annotations"), {".xml"}):
                return "voc"
    return "yolo"


def _discover_voc_splits(src_dir: str) -> List[Split]:
    """发现 VOC 数据集的分集

    支持三种结构:
        1. src_dir 本身就是 VOC 根（Annotations/ + JPEGImages/），
           若有 ImageSets/Main/*.txt 则按清单拆分集；
        2. 一级子目录是 VOC 根（如 src_dir/VOC2007/Annotations/）；
        3. 用 _resolve_voc_dirs 智能检测兜底。
    """
    if _looks_like_dir(os.path.join(src_dir, "Annotations"), {".xml"}) \
            or _has_ext(src_dir, {".xml"}):
        ann_dir, img_dir = _resolve_voc_dirs(src_dir)
        return _voc_splits_from_sets(src_dir, ann_dir, img_dir)

    splits: List[Split] = []
    if os.path.isdir(src_dir):
        for name in sorted(os.listdir(src_dir)):
            sub_dir = os.path.join(src_dir, name)
            if not os.path.isdir(sub_dir):
                continue
            if _looks_like_dir(os.path.join(sub_dir, "Annotations"), {".xml"}):
                sub_ann, sub_img = _resolve_voc_dirs(sub_dir)
                splits.append(Split(name, sub_img, sub_ann))
    if splits:
        return splits

    ann_dir, img_dir = _resolve_voc_dirs(src_dir)
    return [Split(None, img_dir, ann_dir)]


def _voc_splits_from_sets(src_dir: str, ann_dir: str, img_dir: str) -> List[Split]:
    """按 ImageSets/Main/*.txt 把 VOC 标注拆成多个分集"""
    sets_dir = os.path.join(src_dir, "ImageSets", "Main")
    if not os.path.isdir(sets_dir):
        return [Split(None, img_dir, ann_dir)]

    splits: List[Split] = []
    taken: Set[str] = set()
    for name in sorted(os.listdir(sets_dir)):
        if not name.lower().endswith(".txt"):
            continue
        token = os.path.splitext(name)[0]
        stems = set()
        try:
            with open(os.path.join(sets_dir, name), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        stems.add(line)
        except OSError:
            continue
        stems -= taken
        if stems:
            splits.append(Split(token, img_dir, ann_dir, stems))
            taken |= stems

    if not splits:
        return [Split(None, img_dir, ann_dir)]

    # 未列入任何清单的标注单独归为一组
    all_stems = {os.path.splitext(f)[0] for f in os.listdir(ann_dir)
                 if f.lower().endswith(".xml")}
    rest = all_stems - taken
    if rest:
        splits.append(Split(None, img_dir, ann_dir, rest))
    return splits


def _read_image_size(image_path: str) -> Tuple[int, int]:
    """只读文件头获取图片尺寸（已按 EXIF 方向修正）

    只读文件头不解码像素，因此判断「图片是否够大」非常廉价。
    """
    with Image.open(image_path) as img:
        width, height = img.size
        try:
            orientation = img.getexif().get(EXIF_ORIENTATION, 1)
        except Exception:
            orientation = 1

    if orientation in ROTATED_ORIENTATIONS:
        width, height = height, width
    return width, height


def _read_yolo_boxes(label_path: str, width: int, height: int,
                     classes: List[str]) -> Tuple[List[SourceBox], int]:
    """读取 YOLO 归一化标注并还原成像素坐标

    Returns:
        (框列表, 无法解析的行数)
    """
    boxes: List[SourceBox] = []
    bad_lines = 0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                bad_lines += 1
                continue

            label = classes[class_id] if 0 <= class_id < len(classes) \
                else f"class_{class_id}"
            boxes.append(SourceBox(
                label, class_id,
                (cx - bw / 2.0) * width, (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width, (cy + bh / 2.0) * height,
            ))

    return boxes, bad_lines


def _read_voc_boxes(xml_path: str) -> Optional[Tuple[List[SourceBox], int, int]]:
    """读取 VOC XML 标注

    Returns:
        (框列表, XML 中记录的宽, XML 中记录的高)，解析失败返回 None
    """
    parsed = _parse_voc_xml(xml_path)
    if parsed is None:
        return None

    boxes = [
        SourceBox(obj["label"], None,
                  float(obj["xmin"]), float(obj["ymin"]),
                  float(obj["xmax"]), float(obj["ymax"]))
        for obj in parsed["objects"]
    ]
    return boxes, parsed["width"], parsed["height"]


def _collect_voc_classes(splits: List[Split]) -> List[str]:
    """扫描所有 VOC 标注收集类别名（排序）"""
    classes = set()
    for split in splits:
        if not os.path.isdir(split.labels_dir):
            continue
        for name in os.listdir(split.labels_dir):
            if not name.lower().endswith(".xml"):
                continue
            if split.stems is not None and os.path.splitext(name)[0] not in split.stems:
                continue
            parsed = _parse_voc_xml(os.path.join(split.labels_dir, name))
            if not parsed:
                continue
            for obj in parsed["objects"]:
                if obj["label"]:
                    classes.add(obj["label"])
    return sorted(classes)


def _load_image(image_path: str) -> Image.Image:
    """打开图片并统一成 RGB（应用 EXIF 方向、透明区合成白底）"""
    img = Image.open(image_path)
    transposed = ImageOps.exif_transpose(img)
    if transposed is not None:
        img = transposed

    if img.mode in ("P", "RGBA", "LA"):
        # 透明区域合成到白底，避免存成 JPEG 后变成黑块
        rgba = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    # 方向已经转正，去掉 EXIF 以免保存时被再次旋转
    img.info.pop("exif", None)
    return img


# ---------------------------------------------------------------- 计划


def _plan_split(split: Split, classes: List[str], size: int,
                limit: int, is_voc: bool) -> Tuple[List[CropItem], Dict[str, int]]:
    """扫描一个分集，生成裁剪计划（不写盘）

    Returns:
        (裁剪计划, 统计计数)
    """
    stats = {
        "images": 0, "crops": 0, "boxes": 0, "source_boxes": 0,
        "clipped_boxes": 0, "dropped_boxes": 0, "invalid_boxes": 0,
        "clipped_anchors": 0, "anchor_lost": 0,
        "skipped_small": 0, "skipped_no_label": 0, "skipped_empty_label": 0,
        "skipped_broken": 0, "bad_label_lines": 0, "size_mismatch": 0,
    }
    items: List[CropItem] = []

    if is_voc:
        if split.stems is not None:
            stems = sorted(split.stems)
        elif os.path.isdir(split.labels_dir):
            stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(split.labels_dir))
                     if f.lower().endswith(".xml")]
        else:
            stems = []
        pairs = []
        for stem in stems:
            xml_path = os.path.join(split.labels_dir, stem + ".xml")
            if os.path.isfile(xml_path):
                pairs.append((xml_path, _find_voc_image(stem, split.images_dir)))
    else:
        image_files = _list_image_files(split.images_dir)
        pairs = [
            (os.path.join(split.labels_dir, os.path.splitext(f)[0] + ".txt"),
             os.path.join(split.images_dir, f))
            for f in image_files
        ]

    if limit > 0:
        pairs = pairs[:limit]

    for label_path, image_path in pairs:
        stats["images"] += 1

        if image_path is None or not os.path.isfile(image_path):
            stats["skipped_no_label"] += 1
            continue
        if not os.path.isfile(label_path):
            stats["skipped_no_label"] += 1
            continue

        try:
            width, height = _read_image_size(image_path)
        except Exception:
            stats["skipped_broken"] += 1
            continue

        if width < size or height < size:
            stats["skipped_small"] += 1
            continue

        if is_voc:
            parsed = _read_voc_boxes(label_path)
            if parsed is None:
                stats["skipped_broken"] += 1
                continue
            boxes, xml_w, xml_h = parsed
            if (xml_w, xml_h) != (width, height):
                # XML 里记录的尺寸不可信，一律以真实图片为准
                stats["size_mismatch"] += 1
        else:
            boxes, bad_lines = _read_yolo_boxes(label_path, width, height, classes)
            stats["bad_label_lines"] += bad_lines

        valid_boxes, invalid = _clamp_boxes_to_image(boxes, width, height)
        stats["invalid_boxes"] += invalid
        stats["source_boxes"] += len(valid_boxes)

        if not valid_boxes:
            stats["skipped_empty_label"] += 1
            continue

        for index, anchor in enumerate(valid_boxes):
            ox, oy = _window_offset(anchor, width, height, size)

            kept, dropped, clipped = _transform_boxes(valid_boxes, ox, oy, size)
            if not kept:
                # 参考目标自身在窗口里连 1 像素都不可见，不出图
                stats["anchor_lost"] += 1
                continue

            anchor_line = next(
                (i for i, box in enumerate(kept) if box["src_index"] == index), -1
            )
            if anchor_line < 0:
                stats["anchor_lost"] += 1
                continue

            if (anchor.xmin - ox < 0.0 or anchor.ymin - oy < 0.0
                    or anchor.xmax - ox > size or anchor.ymax - oy > size):
                # 参考目标自身被窗口切到（目标比窗口还大时必然发生）
                stats["clipped_anchors"] += 1

            items.append(CropItem(image_path, index, ox, oy, kept,
                                  anchor_line, dropped, clipped))
            stats["clipped_boxes"] += clipped
            stats["dropped_boxes"] += dropped

    stats["crops"] = len(items)
    stats["boxes"] = sum(len(item.boxes) for item in items)
    return items, stats


def _find_voc_image(stem: str, images_dir: str) -> Optional[str]:
    """在图片目录中按主干名查找图片（自动匹配扩展名）"""
    if not os.path.isdir(images_dir):
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    # 忽略大小写再试一次
    for name in os.listdir(images_dir):
        if os.path.splitext(name)[0] == stem:
            return os.path.join(images_dir, name)
    return None


# ---------------------------------------------------------------- 可视化


def _write_comparison(orig_img: Image.Image, crop_img: Image.Image,
                      item: CropItem, boxes: List[SourceBox],
                      class_colors: Dict[str, Tuple[int, int, int]],
                      output_path: str, size: int) -> bool:
    """生成「原图（标出裁剪窗口）/ 裁剪结果」左右对比图"""
    orig_w, orig_h = orig_img.size

    # 左面板按比例缩到最大宽度内（原图两边都 >= size，因此 scale <= 1）
    scale = min(size / float(orig_h), MAX_PANEL_W / float(orig_w))
    left_w = max(1, int(round(orig_w * scale)))
    left_h = max(1, int(round(orig_h * scale)))
    if (left_w, left_h) == (orig_w, orig_h):
        left = orig_img.copy()
    else:
        left = orig_img.resize((left_w, left_h), Image.LANCZOS)

    title_h = 24
    gap = 8
    content_h = max(left_h, size)
    canvas = Image.new("RGB", (left_w + gap + size, title_h + content_h), (32, 32, 32))

    left_top = title_h + (content_h - left_h) // 2
    crop_top = title_h + (content_h - size) // 2
    canvas.paste(left, (0, left_top))
    canvas.paste(crop_img, (left_w + gap, crop_top))

    draw = ImageDraw.Draw(canvas)
    font = load_visualization_font(14)

    # 左面板：原图上的所有标注框
    draw_bboxes(draw, [
        {
            "label": box.label,
            "xmin": box.xmin * scale,
            "ymin": box.ymin * scale + left_top,
            "xmax": box.xmax * scale,
            "ymax": box.ymax * scale + left_top,
        }
        for box in boxes
    ], class_colors, font)

    # 左面板：裁剪窗口的位置（最后画，压在最上层）
    win_x1 = item.ox * scale
    win_y1 = item.oy * scale + left_top
    win_x2 = (item.ox + size) * scale
    win_y2 = (item.oy + size) * scale + left_top
    for offset in range(2):
        draw.rectangle(
            [win_x1 - offset, win_y1 - offset, win_x2 + offset, win_y2 + offset],
            outline=(255, 64, 64)
        )

    # 右面板：裁剪图上的标注框
    draw_bboxes(draw, [
        {
            "label": box["label"],
            "xmin": box["xmin"] + left_w + gap,
            "ymin": box["ymin"] + crop_top,
            "xmax": box["xmax"] + left_w + gap,
            "ymax": box["ymax"] + crop_top,
        }
        for box in item.boxes
    ], class_colors, font)

    # 分隔线与标题
    draw.rectangle([left_w + gap // 2 - 1, title_h, left_w + gap // 2, title_h + content_h],
                   fill=(90, 90, 90))
    if font:
        draw.text((6, 4),
                  f"original {orig_w}x{orig_h}  window=({item.ox},{item.oy}) {size}x{size}",
                  fill=(230, 230, 230), font=font)
        draw.text((left_w + gap + 6, 4), f"crop {size}x{size}",
                  fill=(230, 230, 230), font=font)

    try:
        canvas.save(output_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 执行


def _execute_split(items: List[CropItem], split: Split, cfg: Dict[str, Any],
                   classes: List[str], class_colors: Dict[str, Tuple[int, int, int]],
                   digits: int, counters: Dict[str, int]
                   ) -> Tuple[Dict[str, int], List[Tuple[CropItem, str]]]:
    """执行一个分集的裁剪：写出图片、标注与可视化

    Returns:
        (统计计数, [(裁剪计划项, 输出主干名)] —— 只含真正写出的项)
    """
    is_voc = cfg["is_voc"]
    size = cfg["size"]
    out_root = cfg["output_dir"]

    if is_voc:
        out_images_dir = make_output_dir(os.path.join(out_root, "JPEGImages"))
        out_labels_dir = make_output_dir(os.path.join(out_root, "Annotations"))
    else:
        out_images_dir = make_output_dir(os.path.join(out_root, "images", split.token)
                                         if split.token else os.path.join(out_root, "images"))
        out_labels_dir = make_output_dir(os.path.join(out_root, "labels", split.token)
                                         if split.token else os.path.join(out_root, "labels"))

    viz_dir = make_output_dir(os.path.join(out_root, "visualizations")) \
        if cfg["visualizations"] else None
    cmp_dir = make_output_dir(os.path.join(out_root, "comparisons")) \
        if cfg["comparisons"] else None

    ext = cfg["ext"]
    written: List[Tuple[CropItem, str]] = []
    stats = {"crops": 0, "visualizations": 0, "comparisons": 0}
    index = cfg["start"]

    # 同一张原图的多个裁剪图共用一次解码
    grouped: Dict[str, List[CropItem]] = {}
    for item in items:
        grouped.setdefault(item.src_image, []).append(item)

    for src_image, its_items in grouped.items():
        try:
            image = _load_image(src_image)
        except Exception:
            counters["skipped_broken"] += len(its_items)
            continue

        # 对比图左侧需要原图坐标系下的框，每张原图只读一次
        source_boxes: List[SourceBox] = []
        if cmp_dir:
            try:
                source_boxes = _load_source_boxes(src_image, cfg, split)
            except Exception:
                source_boxes = []

        for item in its_items:
            stem = _new_stem(cfg["prefix"], split.token, index, digits)
            index += 1

            crop = image.crop((item.ox, item.oy, item.ox + size, item.oy + size))
            out_image = os.path.join(out_images_dir, stem + ext)
            if os.path.exists(out_image):
                counters["overwritten"] += 1
            _save_image(crop, out_image, cfg)

            if is_voc:
                out_label = os.path.join(out_labels_dir, stem + ".xml")
                xml = _create_voc_xml({"imageWidth": size, "imageHeight": size},
                                      os.path.basename(out_image), item.boxes)
                with open(out_label, "w", encoding="utf-8") as f:
                    f.write(xml)
            else:
                out_label = os.path.join(out_labels_dir, stem + ".txt")
                with open(out_label, "w", encoding="utf-8") as f:
                    for box in item.boxes:
                        cx, cy, bw, bh = yolo_bbox(
                            box["xmin"], box["ymin"], box["xmax"], box["ymax"],
                            size, size
                        )
                        f.write(f"{box['class_id']} {cx:.6f} {cy:.6f} "
                                f"{bw:.6f} {bh:.6f}\n")

            written.append((item, stem))

            if viz_dir and generate_bbox_visualization_from_image(
                    crop, item.boxes, class_colors,
                    os.path.join(viz_dir, stem + ".jpg")):
                stats["visualizations"] += 1

            if cmp_dir:
                if _write_comparison(image, crop, item, source_boxes, class_colors,
                                     os.path.join(cmp_dir, stem + ".jpg"), size):
                    stats["comparisons"] += 1

            stats["crops"] += 1

        image.close()

    if is_voc and written:
        sets_dir = make_output_dir(os.path.join(out_root, "ImageSets", "Main"))
        with open(os.path.join(sets_dir, f"{split.token or 'train'}.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(stem for _, stem in written) + "\n")

    return stats, written


def _load_source_boxes(src_image: str, cfg: Dict[str, Any],
                       split: Split) -> List[SourceBox]:
    """读取某张原图的标注并钳到图片范围内（对比图左侧需要原图坐标）"""
    stem = os.path.splitext(os.path.basename(src_image))[0]

    if cfg["is_voc"]:
        parsed = _read_voc_boxes(os.path.join(split.labels_dir, stem + ".xml"))
        if parsed is None:
            return []
        boxes, width, height = parsed
    else:
        width, height = _read_image_size(src_image)
        boxes, _ = _read_yolo_boxes(
            os.path.join(split.labels_dir, stem + ".txt"),
            width, height, cfg["classes"]
        )

    return _clamp_boxes_to_image(boxes, width, height)[0]


def _save_image(image: Image.Image, output_path: str, cfg: Dict[str, Any]) -> None:
    """保存裁剪图（JPEG 可指定质量）"""
    if output_path.lower().endswith((".jpg", ".jpeg")):
        image.save(output_path, quality=cfg["quality"])
    else:
        image.save(output_path)


# ---------------------------------------------------------------- 元文件


def _write_crop_map(output_dir: str, records: List[Dict[str, Any]]) -> Optional[str]:
    """写出裁剪映射表（TSV），记录每张裁剪图的窗口位置与参考目标"""
    if not records:
        return None

    map_path = os.path.join(output_dir, "crop_map.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("# dstool crop-dataset 裁剪映射表 (TSV)\n")
        f.write("# stem\tsrc_image\tox\toy\tsize\tanchor_index\tanchor_line"
                "\tanchor_label\tboxes\tdropped\tclipped\tsrc_w\tsrc_h\n")
        for r in records:
            f.write("\t".join(str(v) for v in (
                r["stem"], os.path.basename(r["src_image"]), r["ox"], r["oy"], r["size"],
                r["anchor_index"], r["anchor_line"], r["anchor_label"],
                r["boxes"], r["dropped"], r["clipped"], r["src_w"], r["src_h"],
            )) + "\n")
    return map_path


def _write_yolo_meta(output_dir: str, classes: List[str],
                     tokens: List[Optional[str]]) -> None:
    """写出 classes.txt 与 dataset.yaml（YOLO 输出）"""
    with open(os.path.join(output_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for name in classes:
            f.write(f"{name}\n")

    with open(os.path.join(output_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write("# YOLO 数据集配置文件\n")
        f.write("# 由 dstool crop-dataset 自动生成\n\n")
        f.write(f"path: {os.path.abspath(output_dir).replace(os.sep, '/')}\n")
        for token in ("train", "val", "test"):
            suffix = f"/{token}" if token in tokens else ""
            f.write(f"{token}: images{suffix}\n")
        f.write(f"\nnc: {len(classes)}\n")
        if classes:
            f.write("names:\n")
            for name in classes:
                f.write(f"  - {name}\n")


def _is_within(path: str, parent: str) -> bool:
    """判断 path 是否位于 parent 目录内部"""
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)]) \
            == os.path.abspath(parent)
    except ValueError:
        # 不同盘符
        return False


# ---------------------------------------------------------------- 入口


def convert_crop_dataset(
    src_dir: str,
    prefix: str,
    output_dir: Optional[str] = None,
    size: int = DEFAULT_SIZE,
    start: int = 1,
    digits: int = 4,
    quality: int = DEFAULT_QUALITY,
    ext: str = "jpg",
    visualizations: bool = True,
    comparisons: bool = True,
    limit: int = 0,
) -> Dict[str, Any]:
    """按目标把数据集裁剪成固定尺寸的小图

    一张原图有 N 个目标就生成 N 张 size×size 的图，命名格式为
    {前缀}_{分集}_{序号}。窗口按「相对位置等比滑动」定位，保证目标中心在
    裁剪图中的相对位置与原图一致。

    Args:
        src_dir: 数据集根目录（YOLO 或 VOC，自动识别）
        prefix: 输出文件名前缀（如 six-axis）
        output_dir: 输出目录
        size: 裁剪窗口边长（正方形）
        start: 每个分集的起始序号
        digits: 序号位数（不足补零，超出自动加宽）
        quality: JPEG 质量 1-100
        ext: 输出图片扩展名（jpg / png）
        visualizations: 是否生成「裁剪图 + 标注框」可视化
        comparisons: 是否生成「原图 / 裁剪结果」对比图
        limit: 每个分集只处理前 N 张原图（0 = 不限）

    Returns:
        包含裁剪统计信息的字典
    """
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    empty = {
        "format": None, "total_images": 0, "total_crops": 0, "total_labels": 0,
        "total_boxes": 0, "classes": [], "splits": {}, "output_dir": None,
        "skipped_small": 0, "skipped_no_label": 0, "skipped_empty_label": 0,
        "skipped_broken": 0, "crop_map": None, "overwritten": 0,
        "visualizations": 0, "comparisons": 0,
    }

    if not os.path.isdir(src_dir):
        print(f"错误: 目录不存在 - {src_dir}")
        return empty
    if not prefix:
        print("错误: 前缀不能为空")
        return empty
    if size < 1:
        print("错误: 裁剪尺寸需为正整数")
        return empty
    if start < 1 or digits < 1:
        print("错误: start 与 digits 需为正整数")
        return empty
    if ext not in IMAGE_EXTENSIONS:
        print(f"错误: 仅支持输出 {', '.join(IMAGE_EXTENSIONS)} 格式")
        return empty
    if output_dir is None:
        print("错误: 需要提供输出目录")
        return empty

    src_dir = os.path.abspath(src_dir)
    output_dir = os.path.abspath(output_dir)
    if _is_within(output_dir, src_dir):
        print(f"错误: 输出目录不能位于源数据集内部 - {output_dir}")
        print("  否则重跑时会把自己的产物当成输入")
        return empty

    prefix = sanitize_prefix(prefix)
    if not prefix:
        print("错误: 前缀为空或只包含非法字符")
        return empty

    # ---- 第1步：识别格式与分集 ----
    fmt = _detect_format(src_dir)
    is_voc = fmt == "voc"
    if is_voc:
        splits = _discover_voc_splits(src_dir)
    else:
        splits = [Split(token, images_dir, labels_dir)
                  for token, images_dir, labels_dir in _discover_splits(src_dir)]

    if not splits:
        print(f"错误: 在 {src_dir} 中未找到可裁剪的数据结构")
        return empty

    if is_voc:
        classes = _collect_voc_classes(splits)
    else:
        collected: List[str] = []
        for split in splits:
            for name in _get_dataset_classes(src_dir, split.labels_dir):
                if name not in collected:
                    collected.append(name)
        classes = sorted(collected)

    class_colors = get_class_colors(classes)
    make_output_dir(output_dir)

    cfg = {
        "is_voc": is_voc, "size": size, "output_dir": output_dir, "prefix": prefix,
        "start": start, "quality": quality, "ext": ext,
        "visualizations": visualizations, "comparisons": comparisons,
        "classes": classes,
    }
    counters = {"overwritten": 0, "skipped_broken": 0}

    # ---- 第2步：逐分集计划 + 执行 ----
    split_stats: Dict[str, Dict[str, int]] = {}
    records: List[Dict[str, Any]] = []
    tokens: List[Optional[str]] = []
    total_images = total_crops = total_boxes = 0
    total_labels = 0
    viz_count = cmp_count = 0

    for split in splits:
        name = split.token or "全部"
        print(f"扫描 [{name}] ...")
        items, plan_stats = _plan_split(split, classes, size, limit, is_voc)

        if items:
            # 序号位数按本分集的裁剪图数量自动加宽
            split_digits = max(digits, len(str(start + len(items) - 1)))
            exec_stats, written = _execute_split(items, split, cfg, classes,
                                                 class_colors, split_digits, counters)
            for item, stem in written:
                try:
                    src_w, src_h = _read_image_size(item.src_image)
                except Exception:
                    src_w = src_h = 0
                records.append({
                    "stem": stem, "src_image": item.src_image, "ox": item.ox,
                    "oy": item.oy, "size": size, "anchor_index": item.src_index,
                    "anchor_line": item.anchor_line,
                    "anchor_label": item.boxes[item.anchor_line]["label"],
                    "boxes": len(item.boxes), "dropped": item.n_dropped,
                    "clipped": item.n_clipped, "src_w": src_w, "src_h": src_h,
                })
            viz_count += exec_stats["visualizations"]
            cmp_count += exec_stats["comparisons"]
            tokens.append(split.token)
            written_count = len(written)
        else:
            written_count = 0

        total_images += plan_stats["images"]
        total_crops += written_count
        total_labels += written_count
        total_boxes += plan_stats["boxes"]
        for key in ("skipped_small", "skipped_no_label", "skipped_empty_label",
                    "skipped_broken"):
            counters[key] = counters.get(key, 0) + plan_stats[key]

        split_stats[name] = {
            "images": plan_stats["images"],
            "crops": written_count,
            "boxes": plan_stats["boxes"],
            "source_boxes": plan_stats["source_boxes"],
            "clipped_boxes": plan_stats["clipped_boxes"],
            "dropped_boxes": plan_stats["dropped_boxes"],
            "invalid_boxes": plan_stats["invalid_boxes"],
            "clipped_anchors": plan_stats["clipped_anchors"],
            "anchor_lost": plan_stats["anchor_lost"],
            "bad_label_lines": plan_stats["bad_label_lines"],
            "size_mismatch": plan_stats["size_mismatch"],
            "skipped_small": plan_stats["skipped_small"],
            "skipped_no_label": plan_stats["skipped_no_label"],
            "skipped_empty_label": plan_stats["skipped_empty_label"],
            "skipped_broken": plan_stats["skipped_broken"],
        }

    if total_crops == 0:
        print("错误: 没有生成任何裁剪图，请检查数据集结构与标注")
        return empty

    # ---- 第3步：元文件 ----
    if not is_voc:
        _write_yolo_meta(output_dir, classes, tokens)

    map_path = _write_crop_map(output_dir, records)

    totals: Dict[str, int] = {}
    for stats in split_stats.values():
        for key, value in stats.items():
            if key != "images":
                totals[key] = totals.get(key, 0) + value

    return {
        "format": fmt,
        "total_images": total_images,
        "total_crops": total_crops,
        "total_labels": total_labels,
        "total_boxes": total_boxes,
        "classes": classes,
        "splits": split_stats,
        "output_dir": output_dir,
        "skipped_small": counters.get("skipped_small", 0),
        "skipped_no_label": counters.get("skipped_no_label", 0),
        "skipped_empty_label": counters.get("skipped_empty_label", 0),
        "skipped_broken": counters.get("skipped_broken", 0),
        "crop_map": map_path,
        "overwritten": counters.get("overwritten", 0),
        "visualizations": viz_count,
        "comparisons": cmp_count,
        "clipped_boxes": totals.get("clipped_boxes", 0),
        "dropped_boxes": totals.get("dropped_boxes", 0),
        "invalid_boxes": totals.get("invalid_boxes", 0),
        "clipped_anchors": totals.get("clipped_anchors", 0),
        "anchor_lost": totals.get("anchor_lost", 0),
        "bad_label_lines": totals.get("bad_label_lines", 0),
        "size_mismatch": totals.get("size_mismatch", 0),
    }
