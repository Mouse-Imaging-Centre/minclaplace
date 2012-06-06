# a version of setup.py that builds the cython code

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_laplace", ["cython_laplace.pyx"])]
)
