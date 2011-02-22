cimport numpy

from blas cimport ddot
ctypedef numpy.float64_t reals #typedef_for easier reedding

cdef extern from "math.h":
    double c_sin "sin"(double)

def sin(double x):
    return c_sin(x)

def sqrt(double x):
    return inl_sqrt(x)

cpdef dot(numpy.ndarray[reals,ndim = 1] x, numpy.ndarray[reals,ndim = 1] y):
    return ddot(x.shape[0],<reals*>x.data,x.strides[0] // sizeof(reals), <reals*>y.data,y.strides[0] // sizeof(reals))

cdef class Function:
    cpdef double evaluate(self, double x) except *:
        return 0
