from setuptools import setup, find_packages

setup(
    name="retinaface",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g., 'numpy>=1.18.5', 'torch>=1.0'
    ],
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
    python_requires=">=3.6",
)
