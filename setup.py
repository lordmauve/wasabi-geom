from setuptools import setup, find_packages


setup(
    name='wasabi.geom',
    version='0.1.2',
    description="2D vector, line and polygon classes, and a spatial hash implementation",
    long_description=open('README.rst').read(),
    author='Daniel Pope',
    author_email='mauve@mauveweb.co.uk',
    url='https://bitbucket.org/lordmauve/wasabi-geom',
    packages=find_packages(),
    namespace_packages=['wasabi'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2 :: Only',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
