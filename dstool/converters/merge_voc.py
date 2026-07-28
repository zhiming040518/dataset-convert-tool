"""合并多个 VOC 数据集（训练/验证/测试）为一个完整数据集"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from dstool.utils import make_output_dir


def _list_xml_files(annotations_dir: str) -> List[str]:
    """列出 Annotations 目录下所有 XML 文件"""
    result = []
    if not os.path.isdir(annotations_dir):
        return result
    for f in sorted(os.listdir(annotations_dir)):
        if f.lower().endswith(".xml"):
            result.append(f)
    return result


# 常见图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _detect_dir_by_ext(src_dir: str, exts: set, min_ratio: float = 0.3) -> Optional[str]:
    """在 src_dir 的子目录中查找以指定扩展名为主的目录。

    当标准目录名（如 Annotations/、JPEGImages/）不存在时，通过文件类型占比自动判断。

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


def _resolve_voc_dirs(src_dir: str) -> Tuple[str, str]:
    """解析 VOC 数据集的标注和图片目录。

    优先使用标准命名（Annotations/、JPEGImages/），
    失败时通过文件类型自动检测。
    确保两者不指向同一目录。

    Returns:
        (annotations_dir, images_dir) 元组
    """
    ann_dir = os.path.join(src_dir, "Annotations")
    img_dir = os.path.join(src_dir, "JPEGImages")

    ann_exists = os.path.isdir(ann_dir)
    img_exists = os.path.isdir(img_dir)

    if ann_exists and img_exists:
        return ann_dir, img_dir

    if not ann_exists:
        detected = _detect_dir_by_ext(src_dir, {".xml"})
        if detected:
            print(f"  [自动检测] 标注目录: {os.path.basename(detected)}/"
                  f" (标准名 'Annotations/' 未找到)")
            ann_dir = detected

    if not img_exists:
        detected = _detect_dir_by_ext(src_dir, IMG_EXTS)
        if detected:
            if os.path.abspath(detected) != os.path.abspath(ann_dir):
                print(f"  [自动检测] 图片目录: {os.path.basename(detected)}/"
                      f" (标准名 'JPEGImages/' 未找到)")
                img_dir = detected

    return ann_dir, img_dir


def convert_merge_voc(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    output_dir: str = "merged_VOC",
) -> Dict[str, Any]:
    """合并训练/验证/测试三个 VOC 数据集为一个完整数据集

    每个输入目录应为 VOC 格式（含 Annotations/、JPEGImages/）。

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

    active_splits = []
    for name, d in splits:
        if d and os.path.isdir(d):
            active_splits.append((name, os.path.abspath(d)))

    if not active_splits:
        print("错误: 至少需要提供一个有效的输入目录")
        return {"total_xml": 0, "total_images": 0, "splits": {}}

    output_dir = make_output_dir(output_dir)
    ann_dir = make_output_dir(os.path.join(output_dir, "Annotations"))
    jpeg_dir = make_output_dir(os.path.join(output_dir, "JPEGImages"))
    sets_dir = make_output_dir(os.path.join(output_dir, "ImageSets", "Main"))

    stats: Dict[str, Dict] = {}
    total_xml = 0
    total_images = 0
    # 文件名冲突跟踪
    used_basenames: set = set()

    for split_name, src_dir in active_splits:
        src_ann, src_jpeg = _resolve_voc_dirs(src_dir)

        xml_files = _list_xml_files(src_ann)

        # 生成该分集的文件名列表（不含后缀，防冲突加前缀）
        file_names: List[str] = []
        # 构建源图片索引：{原名（不含后缀）: 扩展名}
        src_img_index: Dict[str, str] = {}
        if os.path.isdir(src_jpeg):
            for f in os.listdir(src_jpeg):
                n, e = os.path.splitext(f)
                if e.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                    src_img_index[n] = e

        for xml_file in xml_files:
            base = os.path.splitext(xml_file)[0]

            # 复制 XML（防文件名冲突加前缀）
            if base.lower() in used_basenames:
                dst_xml_name = f"{split_name}_{xml_file}"
                dst_base = f"{split_name}_{base}"
            else:
                dst_xml_name = xml_file
                dst_base = base
            used_basenames.add(dst_base.lower())
            dst_xml_path = os.path.join(ann_dir, dst_xml_name)
            if not os.path.exists(dst_xml_path):
                shutil.copy2(os.path.join(src_ann, xml_file), dst_xml_path)
            file_names.append(dst_base)

        # 复制图片：根据 file_names 从源目录匹配
        img_count = 0
        img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
        for fn in file_names:
            dst_img_path = None
            # 如果文件名带了前缀，去掉前缀找原始文件名
            if fn.startswith(f"{split_name}_"):
                orig_fn = fn[len(split_name) + 1:]
            else:
                orig_fn = fn

            # 先在索引中精确匹配
            if orig_fn in src_img_index:
                ext = src_img_index[orig_fn]
                dst_img_path = os.path.join(jpeg_dir, fn + ext)
                if not os.path.exists(dst_img_path):
                    shutil.copy2(os.path.join(src_jpeg, orig_fn + ext), dst_img_path)
                    img_count += 1
            else:
                # 兜底：逐个扩展名尝试
                for ext in img_exts:
                    src_img_path = os.path.join(src_jpeg, orig_fn + ext)
                    dst_img_path = os.path.join(jpeg_dir, fn + ext)
                    if os.path.isfile(src_img_path) and not os.path.exists(dst_img_path):
                        shutil.copy2(src_img_path, dst_img_path)
                        img_count += 1
                        break

        # 写入该分集的 txt
        txt_path = os.path.join(sets_dir, f"{split_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(file_names) + "\n")

        stats[split_name] = {"xml": len(xml_files), "images": img_count}
        total_xml += len(xml_files)
        total_images += img_count
        print(f"  [{split_name}] {len(xml_files)} 个 XML, {img_count} 张图片")

    # 生成 train.txt（汇总所有分集的文件名，供兼容旧脚本使用）
    all_names = []
    for split_name, _ in active_splits:
        txt_path = os.path.join(sets_dir, f"{split_name}.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                all_names.extend(line.strip() for line in f if line.strip())
    if all_names:
        with open(os.path.join(sets_dir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(all_names) + "\n")

    print(f"\n[OK] 合并完成！数据集已保存至: {output_dir}")

    return {
        "total_xml": total_xml,
        "total_images": total_images,
        "splits": stats,
    }
