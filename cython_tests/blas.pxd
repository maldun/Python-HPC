cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX,double *Y, int incY)

cimport numpy

cpdef dot(numpy.ndarray[reals,ndim = 1] x, numpy.ndarray[reals,ndim = 1] y)

cdef class Function:
    cpdef double evaluate(self, double x)
