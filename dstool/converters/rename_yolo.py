"""重命名 YOLO 数据集中的图片与标注文件（按顺序重新编号）

新文件名格式: {前缀}_{分集}_{序号}.{扩展名}，例如 six-axis_train_0001.jpg。
分集名（train/val/test）插在前缀与序号之间，因此不同分集之间不会出现同名文件。
图片与同名标注 .txt 会同步改名，保证 YOLO 的文件名对应关系不被打乱。
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from dstool.converters.merge_yolo import (
    IMG_EXTS,
    _detect_dir_by_ext,
    _list_image_files,
)
from dstool.utils import make_output_dir

# Windows 文件名中不允许出现的字符
INVALID_NAME_CHARS = '<>:"/\\|?*'

# 重命名计划中的一项: (源路径, 目标路径, 类型 image/label)
PlanItem = Tuple[str, str, str]

# 需要跟随数据集一起复制的配置文件
META_FILES = ("classes.txt", "dataset.yaml", "data.yaml")


def sanitize_prefix(prefix: str) -> str:
    """清理前缀中的非法文件名字符

    Args:
        prefix: 用户输入的前缀

    Returns:
        清理后的前缀（可能为空字符串）
    """
    cleaned = prefix.strip()
    for ch in INVALID_NAME_CHARS:
        cleaned = cleaned.replace(ch, "_")
    # Windows 下文件名不能以点或空格结尾
    return cleaned.strip(". ")


def _new_stem(prefix: str, split_token: Optional[str], index: int, digits: int) -> str:
    """生成新的文件名主干（不含扩展名）

    Args:
        prefix: 文件名前缀
        split_token: 分集名称（train/val/test），单层结构时为 None
        index: 序号
        digits: 序号位数（不足补零）

    Returns:
        形如 six-axis_train_0001 的字符串
    """
    serial = str(index).zfill(digits)
    if split_token:
        return f"{prefix}_{split_token}_{serial}"
    return f"{prefix}_{serial}"


def _has_ext(directory: str, exts: set) -> bool:
    """判断目录下是否直接存在指定扩展名的文件"""
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in exts:
            return True
    return False


def _has_images(directory: str) -> bool:
    """判断目录下是否直接存在图片文件"""
    return _has_ext(directory, IMG_EXTS)


def _looks_like_dir(directory: str, exts: set) -> bool:
    """判断目录本身或它的一级子目录中是否存在指定类型的文件

    用于识别 images/ 与 labels/ 这类目录：分集结构下它们本身不含文件，
    真正的文件在 train/、val/ 等子目录里。
    """
    if _has_ext(directory, exts):
        return True
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        sub_dir = os.path.join(directory, name)
        if os.path.isdir(sub_dir) and _has_ext(sub_dir, exts):
            return True
    return False


def _locate_dirs(root: str) -> Tuple[str, str]:
    """定位 root 下的图片目录与标注目录

    优先使用标准名 images/、labels/；否则按扩展名占比检测子目录；
    文件直接放在 root 下时使用 root 本身。返回的路径可能不存在。

    Args:
        root: 数据集（或分集）根目录

    Returns:
        (图片目录, 标注目录)
    """
    images_dir = os.path.join(root, "images")
    if not _looks_like_dir(images_dir, IMG_EXTS):
        detected = _detect_dir_by_ext(root, IMG_EXTS)
        if detected:
            images_dir = detected
        elif _has_images(root):
            images_dir = root

    labels_dir = os.path.join(root, "labels")
    if not _looks_like_dir(labels_dir, {".txt"}):
        detected = _detect_dir_by_ext(root, {".txt"})
        if detected and os.path.abspath(detected) != os.path.abspath(images_dir):
            labels_dir = detected
        elif os.path.abspath(images_dir) == os.path.abspath(root):
            # 图片直接放在 root 下，标注通常也在同一目录
            labels_dir = root

    return images_dir, labels_dir


def _discover_splits(src_dir: str) -> List[Tuple[Optional[str], str, str]]:
    """发现数据集中的分集及其图片/标注目录

    支持三种目录结构:
        1. images/train、images/val + labels/train、labels/val（merge-yolo 输出）
        2. train/images、train/labels + val/images、val/labels（Roboflow 风格）
        3. images/、labels/（或文件直接放在数据集根目录下）

    Args:
        src_dir: 数据集根目录

    Returns:
        [(分集名, 图片目录, 标注目录), ...]，单层结构的分集名为 None
    """
    images_dir, labels_dir = _locate_dirs(src_dir)

    # 结构 1：图片目录下再按分集分子目录
    splits: List[Tuple[Optional[str], str, str]] = []
    if os.path.isdir(images_dir):
        for name in sorted(os.listdir(images_dir)):
            sub_images = os.path.join(images_dir, name)
            if not os.path.isdir(sub_images) or not _has_images(sub_images):
                continue
            sub_labels = os.path.join(labels_dir, name)
            splits.append((name, sub_images, sub_labels))

    if splits:
        if _has_images(images_dir):
            print(f"  警告: {images_dir} 下同时存在图片与分集子目录，"
                  f"根目录下的图片将被跳过")
        return splits

    if _has_images(images_dir):
        # 结构 3：单层
        return [(None, images_dir, labels_dir)]

    # 结构 2：每个分集自带 images/、labels/
    splits = []
    for name in sorted(os.listdir(src_dir)):
        sub_dir = os.path.join(src_dir, name)
        if not os.path.isdir(sub_dir):
            continue
        sub_images, sub_labels = _locate_dirs(sub_dir)
        # 允许该分集没有 labels/（纯推理集），此时只重命名图片
        if _has_images(sub_images):
            splits.append((name, sub_images, sub_labels))
    return splits


def _remap_dir(path: str, src_root: str, dst_root: Optional[str]) -> str:
    """按输出的数据集根目录重定位子目录（原地重命名时原样返回）

    例如 copy 模式下 labels/train 会变为 <输出目录>/labels/train。
    """
    if dst_root is None:
        return path
    return os.path.join(dst_root, os.path.relpath(path, src_root))


def _build_split_plan(
    split_token: Optional[str],
    images_dir: str,
    labels_dir: str,
    prefix: str,
    start: int,
    digits: int,
    src_root: str,
    dst_root: Optional[str],
) -> Tuple[List[PlanItem], int, int]:
    """为一个分集生成重命名计划

    Args:
        split_token: 分集名称
        images_dir: 该分集的图片目录
        labels_dir: 该分集的标注目录
        prefix: 文件名前缀
        start: 起始序号
        digits: 序号位数
        src_root: 源数据集根目录
        dst_root: 目标数据集根目录，原地重命名时为 None

    Returns:
        (重命名计划, 图片数, 无对应图片的标注数)
    """

    def dst_path(src_file: str) -> str:
        """把源文件路径映射到目标目录下的同名新文件"""
        return os.path.join(
            _remap_dir(os.path.dirname(src_file), src_root, dst_root),
            os.path.basename(src_file),
        )

    image_files = _list_image_files(images_dir)
    if not image_files:
        return [], 0, 0

    # 序号超出位数时自动加宽，避免出现序号被截断的错觉
    digits = max(digits, len(str(start + len(image_files) - 1)))

    items: List[PlanItem] = []
    matched_labels = set()

    for i, img_file in enumerate(image_files):
        stem, ext = os.path.splitext(img_file)
        new_stem = _new_stem(prefix, split_token, start + i, digits)

        items.append((
            os.path.join(images_dir, img_file),
            dst_path(os.path.join(images_dir, new_stem + ext)),
            "image",
        ))

        # YOLO 靠同名 txt 与图片对应，标注文件必须同步改名
        src_label = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(src_label):
            matched_labels.add(stem + ".txt")
            items.append((
                src_label,
                dst_path(os.path.join(labels_dir, new_stem + ".txt")),
                "label",
            ))

    # 统计没有对应图片的标注文件（无法确定序号，保持原名不动）
    orphans = 0
    if os.path.isdir(labels_dir):
        for name in os.listdir(labels_dir):
            if name.lower().endswith(".txt") and name not in matched_labels:
                orphans += 1

    return items, len(image_files), orphans


def _find_conflicts(items: List[PlanItem]) -> List[str]:
    """检查目标文件名是否被计划外的文件占用

    Args:
        items: 重命名计划

    Returns:
        冲突的目标路径列表
    """
    sources = {os.path.abspath(src) for src, _, _ in items}
    conflicts = []
    seen = set()

    for _src, dst, _kind in items:
        if dst in seen:
            # 两个文件被映射到同一个目标名
            conflicts.append(dst)
            continue
        seen.add(dst)
        # 目标已存在且不属于本次计划 → 会被覆盖，属于冲突
        if os.path.exists(dst) and os.path.abspath(dst) not in sources:
            conflicts.append(dst)

    return conflicts


def _is_inside(path: str, parent: str) -> bool:
    """判断 path 是否位于 parent 的 images/ 或 labels/ 目录内部"""
    for name in ("images", "labels"):
        sub_dir = os.path.abspath(os.path.join(parent, name))
        try:
            if os.path.commonpath([os.path.abspath(path), sub_dir]) == sub_dir:
                return True
        except ValueError:
            # 不同盘符，Windows 下 commonpath 会抛出异常
            continue
    return False


def _temp_path(src: str, index: int) -> str:
    """为两阶段重命名生成一个不与现有文件冲突的临时路径"""
    directory, name = os.path.split(src)
    ext = os.path.splitext(name)[1]
    tmp = os.path.join(directory, f".dstool_tmp_{index}{ext}")
    while os.path.exists(tmp):
        index += 100000
        tmp = os.path.join(directory, f".dstool_tmp_{index}{ext}")
    return tmp


def _apply_inplace(items: List[PlanItem]) -> None:
    """原地重命名（两阶段，避免新旧文件名相互占用）

    阶段1 全部改为临时名，阶段2 再由临时名改为目标名。
    阶段1 中途失败时自动回滚已改名的文件。

    Args:
        items: 重命名计划

    Raises:
        RuntimeError: 重命名失败（已尽力回滚）
    """
    staged: List[Tuple[str, str, str]] = []  # (源路径, 临时路径, 目标路径)
    try:
        for i, (src, dst, _kind) in enumerate(items):
            tmp = _temp_path(src, i)
            os.replace(src, tmp)
            staged.append((src, tmp, dst))
    except OSError as e:
        for src, tmp, _dst in staged:
            try:
                os.replace(tmp, src)
            except OSError:
                pass
        raise RuntimeError(f"重命名失败（已回滚 {len(staged)} 个文件）: {e}")

    for _src, tmp, dst in staged:
        os.replace(tmp, dst)


def _apply_copy(items: List[PlanItem]) -> int:
    """复制到新目录并按新名保存

    Returns:
        覆盖已存在文件的次数
    """
    overwritten = 0
    for src, dst, _kind in items:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            overwritten += 1
        shutil.copy2(src, dst)
    return overwritten


def _update_list_files(
    src_root: str,
    dst_root: str,
    mappings: Dict[str, Dict[str, str]],
) -> List[str]:
    """同步更新 train.txt / val.txt 等图片清单文件中的文件名

    只改写能匹配到本次重命名的行，其余内容原样保留；
    没有匹配到任何文件的清单不会被写出。

    Args:
        src_root: 源数据集根目录
        dst_root: 目标数据集根目录（原地重命名时与 src_root 相同）
        mappings: {分集名: {旧图片文件名: 新图片文件名}}
                  按分集区分，避免不同分集中的同名图片互相串用

    Returns:
        被更新的清单文件路径列表
    """
    # 每个分集的清单文件只使用该分集自己的映射
    candidates: List[Tuple[str, Dict[str, str]]] = []
    for token, mapping in mappings.items():
        for base in (src_root, os.path.join(src_root, "ImageSets", "Main")):
            candidates.append((os.path.join(base, f"{token}.txt"), mapping))

    updated: List[str] = []
    for list_path, mapping in candidates:
        if not os.path.isfile(list_path):
            continue
        stem_map = {os.path.splitext(k)[0]: v for k, v in mapping.items()}

        with open(list_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        changed = 0
        new_lines = []
        for line in lines:
            raw = line.rstrip("\r\n")
            ending = line[len(raw):]
            name = os.path.basename(raw.strip().replace("\\", "/"))

            if name in mapping:
                new_name = mapping[name]
            elif os.path.splitext(name)[0] in stem_map:
                # 清单里写的是不带扩展名的文件名
                new_name = os.path.splitext(stem_map[os.path.splitext(name)[0]])[0]
            else:
                new_lines.append(line)
                continue

            pos = raw.rfind(name)
            new_lines.append(raw[:pos] + new_name + ending)
            changed += 1

        if changed == 0:
            continue

        dst_list = os.path.join(dst_root, os.path.relpath(list_path, src_root))
        os.makedirs(os.path.dirname(dst_list), exist_ok=True)
        with open(dst_list, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        updated.append(dst_list)

    return updated


def _sync_meta_files(src_root: str, dst_root: str) -> List[str]:
    """复制 classes.txt / dataset.yaml 等配置文件到新数据集

    dataset.yaml 中的 path 字段会改写为新的数据集根目录。

    Returns:
        已复制/生成的文件路径列表
    """
    copied: List[str] = []
    new_path = os.path.abspath(dst_root).replace(os.sep, "/")

    for name in META_FILES:
        src_file = os.path.join(src_root, name)
        if not os.path.isfile(src_file):
            continue

        dst_file = os.path.join(dst_root, name)
        if name.endswith((".yaml", ".yml")):
            with open(src_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(dst_file, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip().startswith("path:"):
                        f.write(f"path: {new_path}\n")
                    else:
                        f.write(line)
        else:
            shutil.copy2(src_file, dst_file)

        copied.append(dst_file)

    return copied


def _write_rename_map(
    dst_root: str,
    split_records: List[Tuple[Optional[str], List[PlanItem]]],
) -> Optional[str]:
    """写出重命名映射表（旧名 -> 新名），便于核对与回退

    Args:
        dst_root: 输出数据集根目录
        split_records: [(分集名, 重命名计划), ...]

    Returns:
        映射表路径，没有图片时返回 None
    """
    has_image = any(
        kind == "image" for _, items in split_records for _, _, kind in items
    )
    if not has_image:
        return None

    map_path = os.path.join(dst_root, "rename_map.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("# dstool rename-yolo 重命名映射表\n")
        f.write("# 格式: 旧文件名 -> 新文件名\n\n")
        for token, items in split_records:
            f.write(f"[{token if token else '未分集'}]\n")
            for src, dst, kind in items:
                if kind != "image":
                    continue
                f.write(f"{os.path.basename(src)} -> {os.path.basename(dst)}\n")
            f.write("\n")

    return map_path


def convert_rename_yolo(
    src_dir: str,
    prefix: str,
    output_dir: Optional[str] = None,
    inplace: bool = False,
    start: int = 1,
    digits: int = 4,
) -> Dict[str, Any]:
    """按顺序重命名 YOLO 数据集中的图片与标注文件

    新文件名格式: {前缀}_{分集}_{序号}.{扩展名}，如 six-axis_train_0001.jpg；
    单层结构（无 train/val/test 子目录）时为 {前缀}_{序号}.{扩展名}。
    每个分集独立从 start 开始编号。

    Args:
        src_dir: 数据集根目录（含 images/、labels/）
        prefix: 新文件名前缀（如 six-axis）
        output_dir: 输出目录，inplace=False 时使用
        inplace: True 直接重命名原文件，False 复制到 output_dir
        start: 每个分集的起始序号
        digits: 序号位数（不足补零，序号超出时自动加宽）

    Returns:
        包含重命名统计信息的字典
    """
    if not output_dir and not inplace:
        print("错误: 需要提供输出目录，或指定原地重命名")
        return {"total_images": 0, "total_labels": 0, "splits": {}, "prefix": prefix,
                "mode": "copy", "output_dir": None, "updated_lists": [],
                "map_file": None, "orphan_labels": 0, "overwritten": 0}

    dst_root = os.path.abspath(src_dir) if inplace else os.path.abspath(output_dir)

    empty_result = {
        "total_images": 0,
        "total_labels": 0,
        "splits": {},
        "prefix": prefix,
        "mode": "inplace" if inplace else "copy",
        "output_dir": None if inplace else dst_root,
        "updated_lists": [],
        "map_file": None,
        "orphan_labels": 0,
        "overwritten": 0,
    }

    if not os.path.isdir(src_dir):
        print(f"错误: 目录不存在 - {src_dir}")
        return empty_result

    if not prefix:
        print("错误: 前缀不能为空")
        return empty_result

    if start < 1 or digits < 1:
        print("错误: start 与 digits 需为正整数")
        return empty_result

    if not inplace and _is_inside(dst_root, src_dir):
        print(f"错误: 输出目录不能位于源数据集的 images/ 或 labels/ 内部 - {dst_root}")
        return empty_result

    splits = _discover_splits(src_dir)
    if not splits:
        print(f"错误: 在 {src_dir} 中未找到 YOLO 数据结构（images/ 与 labels/）")
        return empty_result

    # ---- 第1步：为每个分集生成重命名计划 ----
    split_plans: List[Tuple[Optional[str], str, str, List[PlanItem], int]] = []
    all_items: List[PlanItem] = []
    orphan_labels = 0

    for token, images_dir, labels_dir in splits:
        items, img_count, orphans = _build_split_plan(
            token, images_dir, labels_dir, prefix, start, digits,
            src_dir, None if inplace else dst_root
        )
        orphan_labels += orphans
        if not items:
            print(f"  警告: {images_dir} 中没有图片，已跳过")
            continue
        split_plans.append((token, images_dir, labels_dir, items, img_count))
        all_items.extend(items)

    if not all_items:
        print("错误: 没有找到可重命名的图片文件")
        return empty_result

    # ---- 第2步：检查目标文件名冲突 ----
    conflicts = _find_conflicts(all_items)
    if conflicts and inplace:
        print(f"\n错误: {len(conflicts)} 个目标文件名已被占用，为避免覆盖数据已中止:")
        for path in conflicts[:5]:
            print(f"  {path}")
        if len(conflicts) > 5:
            print(f"  ... 其余 {len(conflicts) - 5} 个")
        print("  请更换前缀（-prefix）后重试")
        return empty_result

    # ---- 第3步：执行重命名 ----
    if inplace:
        print(f"开始原地重命名 {len(all_items)} 个文件 ...")
        try:
            _apply_inplace(all_items)
        except RuntimeError as e:
            print(f"错误: {e}")
            return empty_result
        overwritten = 0
    else:
        make_output_dir(dst_root)
        print(f"开始复制 {len(all_items)} 个文件到 {dst_root} ...")
        overwritten = _apply_copy(all_items)

    # ---- 第4步：同步图片清单文件（train.txt / val.txt ...）----
    mappings = {
        token: {
            os.path.basename(src): os.path.basename(dst)
            for src, dst, kind in items
            if kind == "image"
        }
        for token, _images_dir, _labels_dir, items, _count in split_plans
        if token
    }
    updated_lists = _update_list_files(src_dir, dst_root, mappings)

    if not inplace:
        _sync_meta_files(src_dir, dst_root)

    map_file = _write_rename_map(dst_root, [(t, items) for t, _, _, items, _ in split_plans])

    # ---- 第5步：统计 ----
    stats: Dict[str, Dict] = {}
    total_images = 0
    total_labels = 0
    for token, _images_dir, _labels_dir, items, img_count in split_plans:
        label_count = sum(1 for _, _, kind in items if kind == "label")
        stats[token if token else "全部"] = {"images": img_count, "labels": label_count}
        total_images += img_count
        total_labels += label_count

    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "splits": stats,
        "prefix": prefix,
        "mode": "inplace" if inplace else "copy",
        "output_dir": None if inplace else dst_root,
        "updated_lists": updated_lists,
        "map_file": map_file,
        "orphan_labels": orphan_labels,
        "overwritten": overwritten,
    }
