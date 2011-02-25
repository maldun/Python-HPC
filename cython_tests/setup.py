from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include as numpy_include

setup(
    name = "Math Stuff",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("math_stuff", ["math_stuff.pyx"],
                   include_dirs=[numpy_include()], libraries=["m"]),
                   Extension("blas_matvec", ["blas_matvec.pyx"],
                   include_dirs=[numpy_include()], libraries=["cblas"])]
)
