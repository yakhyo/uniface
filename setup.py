from setuptools import setup, find_packages

setup(
    name="retinaface",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "opencv-python>=4.10.0.84",
        "opencv-python-headless>=4.9.0.80",
        "onnx>=1.16.0",
        "onnxruntime-gpu>=1.17.1",
        "requests>=2.32.3",
        "torch>=2.3.1"
    ],
    extras_require={
        "dev": ["pytest"],
    },
    description="A lightweight RetinaFace implementation for face detection.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yakhyokhuja Valikhujaev",
    author_email="yakhyo9696@gmail.com",
    url="https://github.com/yakhyo/retinaface",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
)
