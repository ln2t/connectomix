"""Setup configuration for Connectomix."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from version.py
version_file = Path(__file__).parent / "connectomix" / "core" / "version.py"
version = {}
with open(version_file) as f:
    exec(f.read(), version)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="connectomix",
    version=version["__version__"],
    description="BIDS-compliant functional connectivity analysis from fMRIPrep outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LN2T Lab",
    author_email="connectomix@ln2t.org",
    url="https://github.com/ln2t/connectomix",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "connectomix=connectomix.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="neuroimaging fMRI connectivity BIDS fMRIPrep",
)
