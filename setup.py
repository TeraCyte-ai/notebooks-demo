from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="teracyte-notebooks-utils",
    version="0.2.0",
    description="TeraCyte Notebooks Utils - SDK for data analysis notebooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TeraCyte AI",
    url="https://github.com/TeraCyte-ai/data-overview",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "fsspec>=2021.10.0",
        "adlfs>=2023.1.0",
        "duckdb>=0.8.0",
        "ipywidgets>=7.6.0",
        "plotly>=5.0.0",
        "pyarrow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "bokeh>=2.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="data-analysis, jupyter, notebooks, visualization, sdk",
    include_package_data=True,
    package_data={
        "teracyte_notebooks_utils": ["*.md"],
    },
)