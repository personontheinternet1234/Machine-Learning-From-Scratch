r"""
**Setup for library installation.**

Run this file to install the GardenPy library.

In the library root, run either:
    - pip install .
    - pip3 install .

To install the library using editable mode, run either:
    - pip install -e .
    - pip3 install -e .

Verify installation using either:
    - pip list
    - pip3 list
"""

from setuptools import setup, find_packages

# get readme description
with open("README.md", "r") as f:
    long_desc = f.read()

# setup
setup(
    name='gardenpy',
    version='0.0.9',
    packages=find_packages(),
)
