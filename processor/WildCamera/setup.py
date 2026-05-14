from pathlib import Path
from setuptools import setup, find_packages

# Read the README for a nice long_description on PyPI
this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Packages to include — restricted to folders visible in the repo screenshot
packages = find_packages(
    include=[
        "WildCamera",
        "WildCamera.*",
        # "tools",
        # "tools.*",
        # "splits",
        # "splits.*",
        # "asset",
        # "asset.*",
    ],
)

setup(
    name="wildcamera",
    version="0.1.0",
    description="WildCamera: utilities and models (with image restoration functionality).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ShngJZ",
    license="Apache-2.0",
    packages=packages,
    include_package_data=True,  # together with MANIFEST.in to ship non-Python data (e.g., splits, assets)
    python_requires=">=3.8",
    # Keep the core install lean; add your runtime deps here if you want them auto-installed
    install_requires=[
        # "numpy>=1.20",
        # "opencv-python>=4.5",
        # "torch>=1.10",
        # "tqdm",
    ],
    # Optional entry-points if you later add CLIs in tools/cli.py (uncomment & adjust)
    # entry_points={
    #     "console_scripts": [
    #         "wildcamera=tools.cli:main",
    #     ]
    # },
    # If using torch hub, keeping project importable is enough for `torch.hub.load`
)
