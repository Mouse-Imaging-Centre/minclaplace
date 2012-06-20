minclaplace
===========

All the glories of Laplace's equation for measuring cortical thickness
as initially described in:

Jones SE, Buchbinder BR, Aharon I. Three-dimensional mapping of
cortical thickness using Laplace's equation. Hum Brain Mapp. 2000
Sep;11(1):12-32.

and modified somewhat in:

Lerch JP, Carroll JB, Dorr A, Spring S, Evans AC, Hayden MR, Sled JG,
Henkelman RM. Cortical thickness measured from MRI in the YAC128 mouse
model of Huntington's disease. Neuroimage. 2008 Jun;41(2):243-51.

Can be used anywhere two boundaries can be defined and the path from
one boundary to the other has some intrinsic meaning. The cerebral
cortex is the classic example, though it has been applied to bone
cortex, plaque thickness, etc., as well.

Usage
-----

_Detailed usage examples to come_

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