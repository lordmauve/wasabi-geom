from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
from Cython.Build import cythonize

class BuildExt(build_ext):
    """Only the Earstrip extension needs C++17; keep wasabigeom compiled as C."""
    def build_extensions(self):
        for extension in self.extensions:
            if extension.name == '_wasabi_earstrip':
                extension.extra_compile_args = (
                    ['/std:c++17'] if self.compiler.compiler_type == 'msvc'
                    else ['-std=c++17']
                )
        super().build_extensions()


setup(
    long_description=Path('README.md').read_text(encoding='utf8'),
    long_description_content_type='text/markdown',
    ext_modules=cythonize([
        Extension('wasabigeom', ['wasabigeom.pyx']),
        Extension('_wasabi_earstrip', ['_wasabi_earstrip.pyx'], language='c++',
                  depends=['earstrip_bridge.hpp', 'vendor/earstrip/earstrip.hpp']),
    ], compiler_directives={'embedsignature': True}),
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
