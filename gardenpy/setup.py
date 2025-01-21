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

If recognition failed, run either:
    - mac: export PYTHONPATH=$(pwd)
    - windows: set PYTHONPATH=%cd%
"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages


def meta(key):
    # metadata extraction
    pattern = rf"^__{key}__ = ['\"]([^'\"]*)['\"]"
    match = re.search(pattern, init, re.M)
    if match:
        return match.group(1)
    raise RuntimeError(f"Attempted invalid match for {key}.")


# get init file
init = Path(os.path.join(os.path.dirname(__file__), '__init__.py')).read_text()

# get readme description
with open('README.md', 'r') as f:
    long_desc = f.read()

# setup
setup(
    name='gardenpy',
    version=meta('version'),
    description=meta('description'),
    long_description=long_desc,
    author=meta('author'),
    author_email=meta('author_email'),
    url=meta('url'),
    download_url=meta('download_url'),
    license=meta('license'),
    license_file=os.path.join(os.path.dirname(__file__), 'LICENCE'),
    packages=find_packages(),
)
