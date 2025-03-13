import os
from setuptools import setup, find_packages

# Read the README file for the long description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="uniface",
    version="0.1.6",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "onnx",
        "onnxruntime",
        "requests",
        "torch",
        "scikit-image"
    ],
    extras_require={
        "dev": ["pytest"],
    },
    description="UniFace: A Comprehensive Library for Face Detection, Recognition, Landmark Analysis, Age, and Gender Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yakhyokhuja Valikhujaev",
    author_email="yakhyo9696@gmail.com",
    url="https://github.com/yakhyo/uniface",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="face detection, face recognition, facial landmark, facial attribute, onnx, opencv, retinaface",
    python_requires=">=3.8",
)
