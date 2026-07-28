"""合并多个 YOLO 数据集（训练/验证/测试）为一个完整数据集"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from dstool.utils import make_output_dir

# 常见图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _read_classes(classes_path: str) -> Optional[List[str]]:
    """从 classes.txt 读取类别列表，不存在返回 None"""
    if not os.path.isfile(classes_path):
        return None
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _infer_classes_from_labels(labels_dir: str) -> List[str]:
    """从 label txt 文件中推断类别（扫描所有 class_id 取最大+1）"""
    max_id = -1
    if not os.path.isdir(labels_dir):
        return []
    for f in os.listdir(labels_dir):
        if not f.lower().endswith(".txt"):
            continue
        fpath = os.path.join(labels_dir, f)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        try:
                            cid = int(float(parts[0]))
                            max_id = max(max_id, cid)
                        except ValueError:
                            pass
        except Exception:
            pass
    if max_id < 0:
        return []
    return [f"class_{i}" for i in range(max_id + 1)]


def _get_dataset_classes(dataset_dir: str, labels_dir: Optional[str] = None) -> List[str]:
    """获取数据集的类别列表：优先 classes.txt，否则从 labels 推断

    Args:
        dataset_dir: 数据集根目录
        labels_dir: 标注目录（可选，默认使用 dataset_dir/labels）
    """
    classes = _read_classes(os.path.join(dataset_dir, "classes.txt"))
    if classes:
        return classes
    lbl_dir = labels_dir if labels_dir else os.path.join(dataset_dir, "labels")
    return _infer_classes_from_labels(lbl_dir)


def _list_image_files(images_dir: str) -> List[str]:
    """列出目录中所有图片文件"""
    result = []
    if not os.path.isdir(images_dir):
        return result
    for f in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            result.append(f)
    return result


def _detect_dir_by_ext(src_dir: str, exts: set, min_ratio: float = 0.3) -> Optional[str]:
    """在 src_dir 的子目录中查找以指定扩展名为主的目录。

    当标准目录名（如 images/、labels/）不存在时，通过文件类型占比自动判断。

    Args:
        src_dir: 数据集根目录
        exts: 目标文件扩展名集合
        min_ratio: 该类型文件在目录中的最低占比（默认 0.3）

    Returns:
        匹配的目录路径，或 None
    """
    if not os.path.isdir(src_dir):
        return None
    best_dir = None
    best_count = 0
    for name in os.listdir(src_dir):
        subdir = os.path.join(src_dir, name)
        if not os.path.isdir(subdir):
            continue
        try:
            files = os.listdir(subdir)
        except (PermissionError, OSError):
            continue
        if not files:
            continue
        match_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in exts)
        ratio = match_count / len(files)
        if ratio >= min_ratio and match_count > best_count:
            best_count = match_count
            best_dir = subdir
    return best_dir


def _resolve_yolo_dirs(src_dir: str) -> Tuple[str, str]:
    """解析 YOLO 数据集的 images 和 labels 目录。

    优先使用标准命名（images/、labels/），
    失败时通过文件类型自动检测。
    确保两者不指向同一目录。

    Returns:
        (images_dir, labels_dir) 元组
    """
    images_dir = os.path.join(src_dir, "images")
    labels_dir = os.path.join(src_dir, "labels")

    img_exists = os.path.isdir(images_dir)
    lbl_exists = os.path.isdir(labels_dir)

    if img_exists and lbl_exists:
        return images_dir, labels_dir

    if not img_exists:
        detected = _detect_dir_by_ext(src_dir, IMG_EXTS)
        if detected:
            print(f"  [自动检测] 图片目录: {os.path.basename(detected)}/"
                  f" (标准名 'images/' 未找到)")
            images_dir = detected

    if not lbl_exists:
        detected = _detect_dir_by_ext(src_dir, {".txt"})
        if detected:
            if os.path.abspath(detected) != os.path.abspath(images_dir):
                print(f"  [自动检测] 标注目录: {os.path.basename(detected)}/"
                      f" (标准名 'labels/' 未找到)")
                labels_dir = detected

    return images_dir, labels_dir


def _remap_labels(src_label_path: str, dst_label_path: str, class_mapping: Dict[int, int]):
    """读取 label 文件，重映射 class_id 后写入新文件

    Args:
        src_label_path: 源 label 文件路径
        dst_label_path: 目标 label 文件路径
        class_mapping: {old_class_id: new_class_id}
    """
    os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
    with open(src_label_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    with open(dst_label_path, "w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                old_id = int(float(parts[0]))
            except ValueError:
                f_out.write(line + "\n")
                continue
            new_id = class_mapping.get(old_id, old_id)
            parts[0] = str(new_id)
            f_out.write(" ".join(parts) + "\n")


def convert_merge_yolo(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    output_dir: str = "merged_dataset",
) -> Dict[str, Any]:
    """合并训练/验证/测试三个 YOLO 数据集为一个完整数据集

    每个输入目录应为 YOLO 格式（含 images/、labels/、classes.txt）。

    Args:
        train_dir: 训练集目录（可选）
        val_dir: 验证集目录（可选）
        test_dir: 测试集目录（可选）
        output_dir: 输出目录

    Returns:
        包含合并统计信息的字典
    """
    splits = [
        ("train", train_dir),
        ("val", val_dir),
        ("test", test_dir),
    ]

    # 过滤有效输入
    active_splits = []
    for name, d in splits:
        if d and os.path.isdir(d):
            active_splits.append((name, os.path.abspath(d)))

    if not active_splits:
        print("错误: 至少需要提供一个有效的输入目录")
        return {"total_images": 0, "total_labels": 0, "classes": [], "splits": {}}

    output_dir = make_output_dir(output_dir)

    # ---- 解析每个分集的目录（标准名称优先，智能检测后备） ----
    split_infos = []
    for split_name, src_dir in active_splits:
        images_dir, labels_dir = _resolve_yolo_dirs(src_dir)
        split_infos.append((split_name, src_dir, images_dir, labels_dir))

    # ---- 第1步：收集所有类别 ----
    all_classes: List[str] = []
    for _, src_dir, _, labels_dir in split_infos:
        for cls_name in _get_dataset_classes(src_dir, labels_dir):
            if cls_name not in all_classes:
                all_classes.append(cls_name)
    all_classes = sorted(all_classes)
    new_class_to_id = {name: i for i, name in enumerate(all_classes)}

    print(f"合并类别 ({len(all_classes)} 类): {', '.join(all_classes)}")

    # ---- 第2步：处理每个分集 ----
    stats: Dict[str, Dict] = {}
    total_images = 0
    total_labels = 0
    for split_name, src_dir, images_dir, labels_dir in split_infos:

        out_img_dir = make_output_dir(os.path.join(output_dir, "images", split_name))
        out_lbl_dir = make_output_dir(os.path.join(output_dir, "labels", split_name))

        # 获取该数据集的类别映射
        src_classes = _get_dataset_classes(src_dir, labels_dir)
        old_to_new: Dict[int, int] = {}
        for old_id, name in enumerate(src_classes):
            if name in new_class_to_id:
                old_to_new[old_id] = new_class_to_id[name]
            # 如果名字不在合并列表中（不太可能发生），保持原 ID

        img_files = _list_image_files(images_dir)
        img_count = 0
        lbl_count = 0

        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(images_dir, img_file)

            # 查找对应的 label 文件
            label_file = None
            for candidate in [base_name + ".txt", img_file.rsplit(".", 1)[0] + ".txt"]:
                candidate_path = os.path.join(labels_dir, candidate)
                if os.path.isfile(candidate_path):
                    label_file = candidate
                    break

            if label_file is None:
                # 没有标注文件也复制图片（纯推理集）
                pass

            # 复制图片
            dst_img_name = img_file
            dst_img_path = os.path.join(out_img_dir, dst_img_name)
            if not os.path.exists(dst_img_path):
                shutil.copy2(img_path, dst_img_path)
            img_count += 1

            # 复制并重映射 label
            if label_file:
                src_lbl_path = os.path.join(labels_dir, label_file)
                dst_lbl_name = os.path.splitext(dst_img_name)[0] + ".txt"
                dst_lbl_path = os.path.join(out_lbl_dir, dst_lbl_name)
                _remap_labels(src_lbl_path, dst_lbl_path, old_to_new)
                lbl_count += 1

        stats[split_name] = {"images": img_count, "labels": lbl_count}
        total_images += img_count
        total_labels += lbl_count
        print(f"  [{split_name}] {img_count} 张图片, {lbl_count} 个标注")

    # ---- 第3步：生成 classes.txt ----
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in all_classes:
            f.write(f"{name}\n")

    # ---- 第4步：生成 dataset.yaml ----
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLO 数据集配置文件\n")
        f.write("# 由 dstool merge-yolo 自动生成\n\n")
        f.write(f"path: {os.path.abspath(output_dir).replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(all_classes)}\n")
        if all_classes:
            f.write("names:\n")
            for name in all_classes:
                f.write(f"  - {name}\n")

    print(f"\n[OK] 合并完成！数据集已保存至: {output_dir}")

    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "classes": all_classes,
        "splits": stats,
    }
