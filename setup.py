from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("wasabigeom.pyx"),
    zip_safe=False,
)
