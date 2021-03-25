from setuptools import setup
from pathlib import Path
from Cython.Build import cythonize

setup(
    long_description=Path('README.md').read_text(encoding='utf8'),
    long_description_content_type='text/markdown',
    ext_modules=cythonize("wasabigeom.pyx", compiler_directives={'embedsignature': True}),
    zip_safe=False,
)
