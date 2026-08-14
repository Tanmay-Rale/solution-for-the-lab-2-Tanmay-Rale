"""
setup.py for the elec5308 package.

Install in editable mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="elec5308",
    version="0.2.0",
    description="ELEC5308 – Intelligent Information Engineering Practice (Lab Utilities)",
    author="ELEC5308 Teaching Team, University of Sydney",
    packages=find_packages(exclude=["tests*", "gradescope*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "comet-ml>=3.35.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
