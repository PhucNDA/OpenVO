# setup.py
from setuptools import setup, find_packages
from pathlib import Path
import os

ROOT = Path(__file__).parent

long_desc = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="metric3d",
    version="0.2.0",
    description="Metric3D: Monocular Metric Depth Estimation",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Yvan Yin et al.",
    license="BSD-2-Clause",
    url="https://github.com/YvanYin/Metric3D",
    python_requires=">=3.8",

    # packages: everything under mono/ (and subpackages) must contain __init__.py
    packages=find_packages(include=["mono", "mono.*"]),

    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # If you have CLIs later, add them here, e.g.:
    # entry_points={"console_scripts": ["metric3d-demo=mono.scripts.demo:main"]},
)
