#!/usr/bin/env python
"""
LiteALPR - Accurate and Efficient General OCR System
"""

import os

from setuptools import find_packages, setup


# Read the contents of README file
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    return ""


# Get version
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "litealpr", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.2"


setup(
    name="litealpr",
    version=get_version(),
    description="Accurate and Efficient General OCR System",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Anh Nguyen",
    author_email="anhnth.25ai@ou.edu.vn",
    url="https://github.com/vn-anhnth/LiteALPR",
    license="GNU Affero General Public License v3.0 (AGPL-3.0)",
    # Package configuration
    packages=find_packages(include=["litealpr", "litealpr.*"]),
    install_requires=[
        "numpy<2.0",
        "opencv-python<=5.0.0.93",
        "PyYAML<=6.0.3",
        "huggingface-hub<=0.31",
        "torch>=1.7.0",
        "torchvision",
        "ultralytics<=8.3.99",
    ],
    extras_require={
        "cpu": ["onnxruntime>=1.13.0"],
        "gpu": ["onnxruntime-gpu>=1.13.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    # Python version requirement
    python_requires=">=3.8",
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Keywords
    keywords="alpr, ocr, automatic-license-plate-recognition, automatic-number-plate-recognition, computer-vision, license-plate-recognition, onnxruntime, plate-recognition, yolov8, python, litealpr, plate-detection, realtime, svtr",
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/vn-anhnth/LiteALPR/issues",
        "Source": "https://github.com/vn-anhnth/LiteALPR",
    },
    # Zip safe
    zip_safe=False,
)
