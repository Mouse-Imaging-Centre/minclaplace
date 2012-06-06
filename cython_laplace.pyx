# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

# compile as follows:
# cython cython_laplace.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o cython_laplace.so cython_laplace.c

# note - there are smarter ways of doing this in distutils.
# also note - uses fused types, so requires at least cython version 0.16

import numpy as np
cimport numpy as np

# some math imports
cdef extern from "math.h":
    double sqrt(double x)
    double floor(double x)

# some type definitions - first for the floats
FDTYPE = np.float64
ctypedef np.float64_t FDTYPE_t

# ... and then for the unsigned bytes
BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t

# and a fused type to let you use either
ctypedef fused binary_or_double:
    np.ndarray[np.float64_t, ndim=3]
    np.ndarray[np.uint8_t, ndim=3]


# a single iteration in solving Laplace's equation
cdef double cythonLaplaceStep(np.ndarray[BDTYPE_t, ndim=3] g,
                              np.ndarray[FDTYPE_t, ndim=3] o):

    # get the dimension info - again, notice the importance of defining types
    cdef int nv0 = g.shape[0]
    cdef int nv1 = g.shape[1]
    cdef int nv2 = g.shape[2]

    cdef int v0, v1, v2

    cdef double convergence = 0.0
    cdef double oldvalue, tmpvalue
    cdef double counter = 0
    cdef unsigned char gridvoxel
    cdef float thirdboundary = 19.9

    # the actual loop - looks identical to the python code
    for v0 in range(nv0):
        for v1 in range(nv1):
            for v2 in range(nv2):
                gridvoxel = g[v0,v1,v2]
                if gridvoxel > 0 and gridvoxel < 10:
                    tmpvalue = 0.0
                    oldvalue = o[v0,v1,v2]
                    counter = 0.0
                    
                    if g[v0+1,v1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += o[v0+1,v1,v2]
                    if g[v0-1,v1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += o[v0-1,v1,v2]
                    if g[v0,v1+1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += o[v0,v1+1,v2]
                    if g[v0,v1-1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += o[v0,v1-1,v2]
                    if g[v0,v1,v2+1] < thirdboundary and nv2 > 1:
                        counter += 1
                        tmpvalue += o[v0,v1,v2+1]
                    if g[v0,v1,v2-1] < thirdboundary and nv2 > 1:
                        counter += 1
                        tmpvalue += o[v0,v1,v2-1]

                    o[v0,v1,v2] = tmpvalue / counter
                    convergence += abs(oldvalue - o[v0,v1,v2])
    return(convergence)


# shamelessly stolen from John Sled's vessel_tracking.py
# this code is not used at the moment, as it's rather slow
def trilinear_interpolant(image, point):
    "trilinear interpolate 3D image block based on 8 neighbours of specified point"

    p = np.floor(point).astype(np.int_)
    u = point - p

    c3 = image[p[0]:p[0]+2, p[1]:p[1]+2, p[2]:p[2]+2]

    # reduce to 2D problem
    d3 = c3[1] - c3[0]
    c2 = c3[0] + u[0]*d3

    # reduce to 1D problem
    d2 = c2[1] - c2[0]
    c1 = c2[0] + u[1]*d2

    # solve 1D problem
    d1 = c1[1] - c1[0]
    value = c1[0] + u[2]*d1

    return value

# shamelessly stolen/adapted code from volume_io
# reimplemented in cython after it turns out that John's python
# interpolant is too slow.
cpdef double fast_trilinear_interpolant(binary_or_double image,
                                        np.ndarray[FDTYPE_t, ndim=1] point):
    cdef int i = <int> floor(point[0])
    cdef int j = <int> floor(point[1])
    cdef int k = <int> floor(point[2])
    cdef double coefs[8]
    
    # fill the coefficients
    coefs[0] = <double> image[ i  , j  , k   ]
    coefs[1] = <double> image[ i  , j  , k+1 ]
    coefs[2] = <double> image[ i  , j+1, k   ]
    coefs[3] = <double> image[ i  , j+1, k+1 ]
    coefs[4] = <double> image[ i+1, j  , k   ]
    coefs[5] = <double> image[ i+1, j  , k+1 ]
    coefs[6] = <double> image[ i+1, j+1, k   ]
    coefs[7] = <double> image[ i+1, j+1, k+1 ]

    # get the offets from the floor of the indices
    cdef double u = point[0] - i
    cdef double v = point[1] - j
    cdef double w = point[2] - k

    # get the four differences in the u direction
    cdef double du00 = coefs[4] - coefs[0]
    cdef double du01 = coefs[5] - coefs[1]
    cdef double du10 = coefs[6] - coefs[2]
    cdef double du11 = coefs[7] - coefs[3]

    # reduce to a 2D problem by interpolating in the u direction
    cdef double c00 = coefs[0] + u * du00
    cdef double c01 = coefs[1] + u * du01
    cdef double c10 = coefs[2] + u * du10
    cdef double c11 = coefs[3] + u * du11

    # get the two differences in the v direction for the 2D problem
    cdef double dv0 = c10 - c00
    cdef double dv1 = c11 - c01

    # reduce 2D to a 1D problem by interpolating in the v direction
    cdef double c0 = c00 + v * dv0
    cdef double c1 = c01 + v * dv1

    # get the 1 difference in the w direction for the 1D problem
    cdef double dw = c1 - c0

    # interpolate in 1D to get the value
    cdef double value = c0 + w * dw
    return(value)

# bilinear interpolant - this time shamelessly stolen from wikipaedia.
# currently assumes that the input is a three-dimensional image where the 
# third dimension has a length of 1.
cpdef double fast_bilinear_interpolant(binary_or_double image, 
                                       np.ndarray[FDTYPE_t, ndim=1] point):
    cdef int i = <int> floor(point[0])
    cdef int j = <int> floor(point[1])
    cdef int k = <int> 0

    cdef double u = point[0] - i
    cdef double v = point[1] - j

    cdef double b1 = image[i,j,0]
    cdef double b2 = image[i+1,j,0] - image[i,j,0]
    cdef double b3 = image[i,j+1,0] - image[i,j,0]
    cdef double b4 = image[i,j,0] - image[i+1,j,0] - image[i,j+1,0] + image[i+1,j+1,0]

    return(b1 + (b2*u) + (b3*v) + (b4*u*v))

cdef np.ndarray[FDTYPE_t, ndim=1] eulerStep(double v0, double v1, double v2,
               double dv0, double dv1, double dv2,
               double h):
    cdef np.ndarray[FDTYPE_t, ndim=1] newpoint = np.zeros(3)
    newpoint[0] = v0 + dv0 * h
    newpoint[1] = v1 + dv1 * h
    newpoint[2] = v2 + dv2 * h
    return(newpoint)

# given the gradients and a voxel (in voxel coordinates), find the streamline
# that goes to the inside and outside boundary
cdef double createStreamline(np.ndarray[BDTYPE_t, ndim=3] g,    # grid
                             np.ndarray[FDTYPE_t, ndim=3] dv0,  # gradient
                             np.ndarray[FDTYPE_t, ndim=3] dv1,  # gradient
                             np.ndarray[FDTYPE_t, ndim=3] dv2,  # gradient
                             double v0, double v1, double v2,   # voxel
                             double h):  # step
    cdef double stream_length, stream_lengthtwo
    cdef double real_line_distance = 0
    cdef double mag
    cdef double grid_position
    cdef double newv0, newv1, newv2
    cdef np.ndarray[FDTYPE_t, ndim=1] point = np.zeros(3)
    cdef np.ndarray[FDTYPE_t, ndim=1] oldpoint = np.zeros(3)

    cdef double h_negative = h * -1

    # get the size of the last dimension - if it's 1, go two-dimensional
    cdef int nv2 = g.shape[2]

    # initialize with input parameters
    oldpoint[0] = v0
    oldpoint[1] = v1
    oldpoint[2] = v2
    if nv2 == 1:
        grid_position = fast_bilinear_interpolant(g, oldpoint)
        newv2 = 0
    else:
        grid_position = fast_trilinear_interpolant(g, oldpoint)

    stream_length = 0
    stream_lengthtwo = 0
    cdef int counter = 0

    # move towards outer surface first
    while grid_position < 9.99:
        if nv2 == 1:
            newv0 = fast_bilinear_interpolant(dv0, oldpoint)
            newv1 = fast_bilinear_interpolant(dv1, oldpoint)
        else:
            newv0 = fast_trilinear_interpolant(dv0, oldpoint)
            newv1 = fast_trilinear_interpolant(dv1, oldpoint)
            newv2 = fast_trilinear_interpolant(dv2, oldpoint)
        mag = newv0*newv0 + newv1*newv1 + newv2*newv2
        
        if mag < 1.0e-6:
            grid_position=10
        else:

            # for some reason eulerStep is not being optimized
            #point = eulerStep(oldpoint[0], oldpoint[1], oldpoint[2],
            #                  newv0, newv1, newv2, h)
            # so do it inline
            point[0] = oldpoint[0] + newv0 * h
            point[1] = oldpoint[1] + newv1 * h
            point[2] = oldpoint[2] + newv2 * h

            stream_length = stream_length + sqrt( (point[0] - oldpoint[0])*(point[0] - oldpoint[0]) + (point[1] - oldpoint[1])*(point[1] - oldpoint[1]) + (point[2]-oldpoint[2])*(point[2]-oldpoint[2]) )

            if nv2 == 1:
                grid_position = fast_bilinear_interpolant(g, point)
            else:
                grid_position = fast_trilinear_interpolant(g, point)
            oldpoint[0] = point[0]
            oldpoint[1] = point[1]
            oldpoint[2] = point[2]
            counter=counter+1
            real_line_distance = sqrt ( (newv0-v0)*(newv0-v0) + 
                                        (newv1-v1)*(newv1-v1) + 
                                        (newv2-v2)*(newv2-v2) )
            if stream_length > (4.0*real_line_distance):
                grid_position = 10.0
        #print "OUTSIDE:", point[0], point[1], point[2], grid_position
    # move towards inner surface
    oldpoint[0] = v0
    oldpoint[1] = v1
    oldpoint[2] = v2
    counter=0
    real_line_distance = 0
    while grid_position > 0.01:
        if nv2 == 1:
            newv0 = fast_bilinear_interpolant(dv0, oldpoint)
            newv1 = fast_bilinear_interpolant(dv1, oldpoint)
        else:
            newv0 = fast_trilinear_interpolant(dv0, oldpoint)
            newv1 = fast_trilinear_interpolant(dv1, oldpoint)
            newv2 = fast_trilinear_interpolant(dv2, oldpoint)

        mag = newv0*newv0 + newv1*newv1 + newv2*newv2
        if mag < 1.0e-6:
            grid_position=0.0
        else:

            # for some reason eulerStep is not being optimized 
            #point = eulerStep(oldpoint[0], oldpoint[1], oldpoint[2],
            #                  newv0, newv1, newv2, (h * -1))
            # so do it inline
            point[0] = oldpoint[0] + newv0 * h_negative
            point[1] = oldpoint[1] + newv1 * h_negative
            point[2] = oldpoint[2] + newv2 * h_negative

            stream_lengthtwo = stream_lengthtwo + sqrt( (point[0] - oldpoint[0])*(point[0] - oldpoint[0]) + (point[1] - oldpoint[1])*(point[1] - oldpoint[1]) + (point[2]-oldpoint[2])*(point[2]-oldpoint[2]) )

            if nv2 == 1:
                grid_position = fast_bilinear_interpolant(g, point)
            else:
                grid_position = fast_trilinear_interpolant(g, point)
            #print "INSIDE:", point[0], point[1], point[2], grid_position, counter
            counter = counter + 1
            oldpoint[0] = point[0]
            oldpoint[1] = point[1]
            oldpoint[2] = point[2]

            real_line_distance = sqrt ( (newv0-v0)*(newv0-v0) + 
                                        (newv1-v1)*(newv1-v1) + 
                                        (newv2-v2)*(newv2-v2) )
            if stream_lengthtwo > (4.0*real_line_distance):
                grid_position = 0.0
    return(stream_length + stream_lengthtwo)


cdef double voxelDistance(np.ndarray[BDTYPE_t, ndim=3] g, # grid
                   int vv0, int vv1, int vv2):     # voxel indices

    cdef int v0, v1, v2
    cdef int nv0 = g.shape[0]
    cdef int nv1 = g.shape[1]
    cdef int nv2 = g.shape[2]
    
    cdef double dist_to_inside = max(nv0, nv1, nv2)
    cdef double dist_to_outside = dist_to_inside

    cdef int current_grid

    cdef double current_distance = 0.0

    for v0 in range(nv0):
       for v1 in range(nv1):
           for v2 in range(nv2):
               current_grid = g[v0,v1,v2]
               if current_grid == 0 or current_grid == 10:
                   current_distance= sqrt( (vv0-v0)*(vv0-v0) + 
                                           (vv1-v1)*(vv1-v1) +
                                           (vv2-v2)*(vv2-v2) )
                   if current_grid == 0 and current_distance < dist_to_inside:
                       dist_to_inside = current_distance
                   elif current_grid == 10 and current_distance < dist_to_outside:
                       dist_to_outside = current_distance
    return(dist_to_inside + dist_to_outside)
               
               
# compute the straight line distance at every voxel
# the third parameter (laplace_initialization) sets the mode:
#   0: output the distance between closest inside and closest outside point
#   1: normalize that distance to lie between the inside and outside grid
def straightLineDistance(np.ndarray[BDTYPE_t, ndim=3] g,    # grid
                         np.ndarray[FDTYPE_t, ndim=3] o,    # output
                         int laplace_initialization):       # work mode
    cdef int v0, v1, v2
    cdef int nv0 = g.shape[0]
    cdef int nv1 = g.shape[1]
    cdef int nv2 = g.shape[2]
    
    for v0 in range(nv0):
        print "In slice:", v0
        for v1 in range(nv1):
            for v2 in range(nv2):
                if g[v0,v1,v2] > 0 and g[v0,v1,v2] < 10:
                    o[v0,v1,v2] = voxelDistance(g, v0, v1, v2)

def computeAllStreamlines(np.ndarray[BDTYPE_t, ndim=3] g,    # grid
                          np.ndarray[FDTYPE_t, ndim=3] o,    # output
                          np.ndarray[FDTYPE_t, ndim=3] dv0,  # gradient
                          np.ndarray[FDTYPE_t, ndim=3] dv1,  # gradient
                          np.ndarray[FDTYPE_t, ndim=3] dv2,  # gradient
                          double h):
    cdef int v0, v1, v2
    cdef int nv0 = g.shape[0]
    cdef int nv1 = g.shape[1]
    cdef int nv2 = g.shape[2]

    for v0 in range(nv0):
        print "In slice:", v0
        for v1 in range(nv1):
            for v2 in range(nv2):
                if g[v0,v1,v2] > 0 and g[v0,v1,v2] < 10:
                    o[v0,v1,v2] = createStreamline(g, dv0, dv1, dv2,
                                                   v0,v1,v2, h)
                else:
                    o[v0,v1,v2] = 0

def computeStreamlinesFromList(np.ndarray[BDTYPE_t, ndim=3] g,    # grid
                               np.ndarray[FDTYPE_t, ndim=2] pointList,
                               np.ndarray[FDTYPE_t, ndim=1] o,    # output
                               np.ndarray[FDTYPE_t, ndim=3] dv0,  # gradient
                               np.ndarray[FDTYPE_t, ndim=3] dv1,  # gradient
                               np.ndarray[FDTYPE_t, ndim=3] dv2,  # gradient
                               double h):
    cdef unsigned long nv0 = pointList.shape[0]
    cdef unsigned long int v0
    for v0 in range(nv0):
        o[v0] = createStreamline(g, dv0, dv1, dv2,
                                 pointList[v0, 0],
                                 pointList[v0, 1],
                                 pointList[v0, 2],
                                 h)

# creates the gradients using the central difference
# this is called after iterateLaplace
def computeGradients(np.ndarray[BDTYPE_t, ndim=3] g, #grid
                     np.ndarray[FDTYPE_t, ndim=3] o, #relaxed equation
                     np.ndarray[FDTYPE_t, ndim=3] dv0, # output gradient
                     np.ndarray[FDTYPE_t, ndim=3] dv1, # output gradient
                     np.ndarray[FDTYPE_t, ndim=3] dv2): # output gradient
    cdef int v0, v1, v2
    cdef int nv0 = g.shape[0]
    cdef int nv1 = g.shape[1]
    cdef int nv2 = g.shape[2]

    cdef double d0, d1, d2, mag
    
    d0 = 0.0

    if nv2 > 1:
        for v0 in range(1, nv0-1):
            for v1 in range(1, nv1-1):
                for v2 in range(1, nv2-1):
                    if g[v0,v1,v2] > 0 and g[v0,v1,v2] < 10:
                        d0 = o[v0+1,v1,v2] - o[v0-1,v1,v2]
                        d1 = o[v0,v1+1,v2] - o[v0,v1-1,v2]
                        d2 = o[v0,v1,v2+1] - o[v0,v1,v2-1]
                        
                        mag = sqrt( (d0*d0) + (d1*d1) + (d2*d2) )
                        dv0[v0,v1,v2] = d0 / mag
                        dv1[v0,v1,v2] = d1 / mag
                        dv2[v0,v1,v2] = d2 / mag
    else:
        v2 = 0
        for v0 in range(1, nv0-1):
            for v1 in range(1, nv1-1):
                if g[v0,v1,v2] > 0 and g[v0,v1,v2] < 10:
                    d0 = o[v0+1,v1,v2] - o[v0-1,v1,v2]
                    d1 = o[v0,v1+1,v2] - o[v0,v1-1,v2]
                    d2 = o[v0,v1,v2+1] - o[v0,v1,v2-1]
                    
                    mag = sqrt( (d0*d0) + (d1*d1) + (d2*d2) )
                    dv0[v0,v1,v2] = d0 / mag
                    dv1[v0,v1,v2] = d1 / mag
                    dv2[v0,v1,v2] = d2 / mag

def iterateLaplace(np.ndarray[BDTYPE_t, ndim=3] g,
                   np.ndarray[FDTYPE_t, ndim=3] o,
                   int max_iterations,
                   double convergence_criteria):

    cdef int i
    cdef double convergence
    cdef double normalize_factor

    # run the first iteration outside of the loop to get the 
    # normalization factor for convergence checking
    normalize_factor = 1.0 / cythonLaplaceStep(g,o)

    for i in range(max_iterations):
        # call a single relaxation step and multiple output by the
        # normalization factor for convergence checking
        convergence = cythonLaplaceStep(g, o) * normalize_factor
        print "iteration", i, ":", convergence

