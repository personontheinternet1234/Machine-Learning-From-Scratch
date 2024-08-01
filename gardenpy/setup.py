# currently the absolute bare minimum amount of information for your computer to recognize this as a library
# we'll actually make this file once gardenpy is functional
# this is like the absolute bare minimum, so it doesn't install any dependencies or anything, it literally just makes the library findable

# to install on computer
# 1. go to the root of this library, so gardenpy
# 2. run either:
#   pip install -e .
#   pip3 install -e .
# 3. go to the root of this repository, so NeuralNet
# 4. run either:
#   mac: export PYTHONPATH=$(pwd)
#   windows: set PYTHONPATH=%cd%

from setuptools import setup, find_packages

setup(
    name='gardenpy',
    version='0.0.8',
    packages=find_packages(),
)
