"""setup.py —— 兼容旧版 pip / --no-build-isolation 场景"""

from setuptools import setup, find_packages

setup(
    name="dstool",
    version="1.0.0",
    description="数据集格式转换工具库 - 支持 LabelMe JSON、VOC XML、YOLO、像素掩码等格式互转",
    author="dstool",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "dstool = dstool.cli:main",
        ],
    },
)
