from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cyvec.pyx"),
    zip_safe=False,
)

