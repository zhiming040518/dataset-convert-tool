"""dstool 命令行接口"""

import argparse
import os
import sys
import textwrap
from typing import Optional, Tuple

# Windows 终端编码兼容处理
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _get_input_path(prompt: str, default: str) -> str:
    """交互式获取输入路径"""
    user_input = input(prompt).strip()
    if not user_input:
        return os.path.abspath(default)
    return os.path.abspath(user_input)


def _get_output_path(prompt: str, default: str) -> str:
    """交互式获取输出路径"""
    user_input = input(prompt).strip()
    if not user_input:
        return os.path.abspath(default)
    return os.path.abspath(user_input)


def _resolve_paths(src: Optional[str], output: Optional[str],
                   src_prompt: str, output_default: str) -> Tuple[str, str]:
    """统一处理路径解析：参数模式优先，否则交互模式

    Args:
        src: -src 参数传入的源路径
        output: -output 参数传入的输出路径
        src_prompt: 交互模式下的输入提示
        output_default: 交互模式下输出路径的默认值

    Returns:
        (source_dir, output_dir) 绝对路径
    """
    interactive = src is None and output is None

    if src:
        source_dir = os.path.abspath(src)
    else:
        source_dir = _get_input_path(src_prompt, os.getcwd())

    if output:
        output_dir = os.path.abspath(output)
    else:
        output_dir = _get_output_path(
            f"请输入输出路径 (留空使用 {output_default}): ",
            output_default
        )

    return source_dir, output_dir


def _cmd_json2voc(args):
    """json2voc 命令处理"""
    from dstool.converters.json2voc import convert_json2voc
    src, out = _resolve_paths(
        args.src, args.output,
        "请输入包含JSON文件的路径 (留空使用当前目录): ",
        os.path.join(os.getcwd(), "VOCdevkit")
    )
    print(f"\n源路径:   {src}")
    print(f"输出路径: {out}\n")
    result = convert_json2voc(src, out)
    print(f"\n[OK] 转换完成！数据集已保存至: {os.path.abspath(out)}")
    print(f"  共处理 {result['total']} 个JSON文件，{result['objects']} 个标注对象")
    print(f"  复制了 {result.get('copied_images', 0)} 张图片")
    print(f"  生成了 {result.get('visualizations', 0)} 张可视化图")
    print(f"  类别: {', '.join(result['classes']) if result['classes'] else '无'}")


def _cmd_json2yolo(args):
    """json2yolo 命令处理"""
    from dstool.converters.json2yolo import convert_json2yolo
    src, out = _resolve_paths(
        args.src, args.output,
        "请输入包含JSON文件的路径 (留空使用当前目录): ",
        os.path.join(os.getcwd(), "YOLO_dataset")
    )
    print(f"\n源路径:   {src}")
    print(f"输出路径: {out}\n")
    result = convert_json2yolo(src, out)
    print(f"\n[OK] 转换完成！数据集已保存至: {os.path.abspath(out)}")
    print(f"  共处理 {result['total']} 个JSON文件，{result['objects']} 个标注对象")
    print(f"  复制了 {result.get('copied_images', 0)} 张图片")
    print(f"  生成了 {result.get('visualizations', 0)} 张可视化图")
    print(f"  类别 ({len(result['classes'])} 类): {', '.join(result['classes']) if result['classes'] else '无'}")


def _cmd_voc2yolo(args):
    """voc2yolo 命令处理"""
    from dstool.converters.voc2yolo import convert_voc2yolo
    src, out = _resolve_paths(
        args.src, args.output,
        "请输入包含XML文件的路径 (留空使用当前目录): ",
        os.path.join(os.getcwd(), "YOLO_from_VOC")
    )
    print(f"\n源路径:   {src}")
    print(f"输出路径: {out}\n")
    result = convert_voc2yolo(src, out)
    print(f"\n[OK] 转换完成！数据集已保存至: {os.path.abspath(out)}")
    print(f"  共处理 {result['total']} 个XML文件，{result['objects']} 个标注对象")
    print(f"  复制了 {result.get('copied_images', 0)} 张图片")
    print(f"  生成了 {result.get('visualizations', 0)} 张可视化图")
    print(f"  类别 ({len(result['classes'])} 类): {', '.join(result['classes']) if result['classes'] else '无'}")


def _cmd_json2mask(args):
    """json2mask 命令处理"""
    from dstool.converters.json2mask import convert_json2mask
    src, out = _resolve_paths(
        args.src, args.output,
        "请输入包含JSON文件的路径 (留空使用当前目录): ",
        os.path.join(os.getcwd(), "Masks")
    )
    print(f"\n源路径:   {src}")
    print(f"输出路径: {out}\n")
    result = convert_json2mask(src, out)
    print(f"\n[OK] 转换完成！掩码已保存至: {os.path.abspath(out)}")
    print(f"  共处理 {result['total']} 个JSON文件，生成 {result['masks']} 个掩码图像")
    print(f"  复制了 {result.get('copied_images', 0)} 张图片")
    print(f"  生成了 {result.get('visualizations', 0)} 张可视化图")
    print(f"  类别 ({len(result['classes'])} 类): {', '.join(result['classes']) if result['classes'] else '无'}")


def main():
    """dstool 主入口"""
    parser = argparse.ArgumentParser(
        prog="dstool",
        description="数据集格式转换工具库 - 支持 LabelMe JSON、VOC XML、YOLO、像素掩码等格式互转",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
使用示例:
  dstool json2voc                       交互模式，按提示输入路径
  dstool json2voc -src ./labels -output ./VOCdevkit   参数模式
  dstool json2yolo -src ./labels -output ./yolo_data
  dstool VOC2yolo -src ./VOCdevkit -output ./yolo_data
  dstool json2mask -src ./labels -output ./masks
        """),
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ---- json2voc / json2VOC ----
    p_json2voc = subparsers.add_parser(
        "json2voc", aliases=["json2VOC"],
        help="将 LabelMe JSON 标注转换为 VOC XML 格式",
        description="将 LabelMe 格式的 JSON 标注文件转换为 Pascal VOC 格式的 XML 标注文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  dstool json2voc\n  dstool json2voc -src ./labels -output ./VOCdevkit",
    )
    p_json2voc.add_argument("-src", type=str, help="包含 JSON 文件的源目录路径")
    p_json2voc.add_argument("-output", type=str, help="输出目录路径")

    # ---- json2yolo ----
    p_json2yolo = subparsers.add_parser(
        "json2yolo",
        help="将 LabelMe JSON 标注转换为 YOLO 格式",
        description="将 LabelMe 格式的 JSON 标注文件转换为 YOLO 格式的 txt 标注文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  dstool json2yolo\n  dstool json2yolo -src ./labels -output ./YOLO_dataset",
    )
    p_json2yolo.add_argument("-src", type=str, help="包含 JSON 文件的源目录路径")
    p_json2yolo.add_argument("-output", type=str, help="输出目录路径")

    # ---- voc2yolo / VOC2yolo ----
    p_voc2yolo = subparsers.add_parser(
        "voc2yolo", aliases=["VOC2yolo"],
        help="将 VOC XML 标注转换为 YOLO 格式",
        description="将 Pascal VOC 格式的 XML 标注文件转换为 YOLO 格式的 txt 标注文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  dstool voc2yolo\n  dstool VOC2yolo -src ./VOCdevkit -output ./YOLO_dataset",
    )
    p_voc2yolo.add_argument("-src", type=str, help="包含 XML 文件的目录路径（自动检测 VOC 目录结构）")
    p_voc2yolo.add_argument("-output", type=str, help="输出目录路径")

    # ---- json2mask ----
    p_json2mask = subparsers.add_parser(
        "json2mask",
        help="将 LabelMe JSON 标注转换为像素掩码",
        description="将 LabelMe 格式的 JSON 标注文件转换为像素级掩码图像（PNG格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  dstool json2mask\n  dstool json2mask -src ./labels -output ./masks",
    )
    p_json2mask.add_argument("-src", type=str, help="包含 JSON 文件的源目录路径")
    p_json2mask.add_argument("-output", type=str, help="输出目录路径")

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 路由到对应的处理函数
    handlers = {
        "json2voc": _cmd_json2voc,
        "json2yolo": _cmd_json2yolo,
        "voc2yolo": _cmd_voc2yolo,
        "json2mask": _cmd_json2mask,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
