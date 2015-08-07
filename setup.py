# the standard setup.py that assumes that the C code has already been generated

from distutils.core import setup
from distutils.extension import Extension

setup(
    name = "minclaplace",
    version = "1.0",
    description = "MINC version of Laplace's equation.",
    author = "Jason Lerch",
    author_email = 'jason.lerch@utoronto.ca',
    url = 'https://github.com/jasonlerch/minclaplace',
    scripts = ["minclaplace", "make_laplace_grid"],
    ext_modules = [Extension("cython_laplace", ["cython_laplace.c"])]
)
