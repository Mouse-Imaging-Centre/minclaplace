minclaplace
===========

Laplace's equation for measuring thickness (or other uses of
streamlines) between two boundaries.

Installing
----------

python setup.py install

(If you want to specify the install location, add the --prefix option, i.e.:
python setup.py install --prefix=/some/directory)

Requires numpy, scipy, and pyminc (version 0.3 or greater).

Compiling
---------

The setup.py script uses the cython generated C code; to regenerate
that C code, i.e. after making changes to cython_laplace.pyx, run:

python setup-cython.py build_ext --inplace

Note that you need cython version 0.16 or greater.