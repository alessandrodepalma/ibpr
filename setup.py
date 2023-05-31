import sys
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

setup(
    name='jax-ibpr',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'torch', 'torchvision', 'scipy', 'jax', 'optax', 'dm-haiku', 'jax_verify',
                      'tensorflow', 'tf2onnx', 'tqdm'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)