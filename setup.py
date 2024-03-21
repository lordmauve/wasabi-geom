from setuptools import setup
from pathlib import Path
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("wasabigeom.pyx", compiler_directives={'embedsignature': True}),
    zip_safe=False,
)
