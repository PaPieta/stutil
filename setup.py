from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="stutil",
    version="0.1.0",
    author="Pawel Pieta",
    author_email="papi@student.dtu.dk",
    description="A Python package with utilities that build on top of the Structure Tensor package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaPieta/structure-tensor-util",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "structure-tensor",
        "tifffile",
        "scikit-image",
        "matplotlib",
        "scmap",
        "PyMaxflow",
        "ipympl"
    ],
)
