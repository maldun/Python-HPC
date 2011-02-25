Cython
======

.. highlight:: cython

Well Cython isn't a part of Python, it is a different language, but
very similar to Python, and in fact it is almost to 90% compatible.
(It is stated that Cython is a superset of Python, but it's currently
under development so there are some features which are not supported yet!)

It first started with the Pyrex project, which allowed to compile
Python to C. The idea was to allow the user to declare C variables and
call C functions within Cython, and make it possible for the C
compiler
to compile the Python like code to fast C code.

Cython has bindings for NumPy, mpi4py and other Python modules to
support scientific computation.

Currently Cython only works on the CPython implementation, but there
are efforts to get it working in IronPython on .Net as well.

I will here give a short tutorial on Cython and demonstrate on an
example how to speed up your NumPy code.

**Important Note** I assume that you are using Linux as operating
system. If you use Windows or an other OS look up the Cython
documentation for specific details! [#]_ 

How to compile your Cython Code
-------------------------------

Sage
""""""""

This is the easiest way. Either write your Cython 
code in a *.spyx* (Sage Pyrex) file, or in the notebook, with the
magic function ``%cython``. 

To use a *.spyx* file simply load it into Sage with the ``load``
command::

  load my_cython_file.spyx

For example I write a short code snippet for an self made
scalar product::

  def my_dot(x,y):
      
      if x.size != y.size
          raise ValueError("Dimension Mismatch")
      
      result = 0
      
      for i in range(x.size):
          result += x[i]*y[i]
      
      return result

I save this in the file my_dot.spyx. Now I call Sage, and cd to the 
directory I saved the file. Now simply call Sage, and type::

  sage: load my_dot.spyx
  Compiling ./my_dot.spyx...

Now the function can be called directly like a normal Python
function::

  sage: from numpy import array
  sage: x = array([1,2,3.])
  sage: y = array([1,0,5])
  sage: my_dot(x,y)
  16.0

A different way would be in the notebook. Simply write in an 
empty notebook cell::

  %cython
  def my_dot(x,y):
      
      if x.size != y.size:
          raise ValueError("Dimension Mismatch")
      
      result = 0
      
      for i in range(x.size):
          result += x[i]*y[i]
      
      return result

Now if you evalute it, the function will be compiled, and you can call
it normally.

Setup files
"""""""""""

The direct approach in Python would be to write a setup file.
First write your code and save it to a *.pyx* file. I use the 
same code as before and write it to *my_dot.pyx*.

Now we use disutils and write a setup.py file, which works similar 
to a make file::

  from distutils.core import setup
  from distutils.extension import Extension
  from Cython.Distutils import build_ext
  
  setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("my_dot", ["my_dot.pyx"])]
  )

Save this as setup.py in the directory where your code file lies.

Now cd to your working directory where the code and setup file is
saved and call it with::

  python setup.py -build_ext --inplace

Then the *.pyx* files will be compiled.
Now you can call it normally in Python (after changing to the working
directory)::

  >>> from my_dot import my_dot

To compile more files, simply put more extensions to the ext_modules
list. I created for example a further file with the name *test.pyx* ::

  from distutils.core import setup
  from distutils.extension import Extension
  from Cython.Distutils import build_ext
  import numpy

  setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("my_dot", ["my_dot.pyx"]),
                     Extension("test", ["test.pyx"])]
  )

**Important:** If you import numpy as C library you have to add
``include_dirs=[numpy.get_include()])`` to the extension. In our
example this would look like this::

  from distutils.core import setup
  from distutils.extension import Extension
  from Cython.Distutils import build_ext
  import numpy

  setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("my_dot", ["my_dot.pyx"],
      include_dirs=[numpy.get_include()])]
  )

  

I state this here, because it is not well documented in
the Cython docu, and I had to search it for long in Cython Mailing
list. How to import modules as C libraries will we see later.


How to use Cython
-----------------

Here we look at the advanced syntax in Cython, and other features in Python.

The `cdef` statment
"""""""""""""""""""

Type declaration
^^^^^^^^^^^^^^^^

``cdef`` is used for C type declaration, and defining C functions.
This can be very useful for speeding up your Python programs.

Let's look at our scalar product again::

  def my_dot(x,y):

      if x.size != y.size
          raise ValueError("Dimension Mismatch")

      result = 0

      for i in range(x.size):
          result += x[i]*y[i]

      return result

The counter variables cost a lot of efficiency because the program has
to check first, what it recieves, because in Python ``i`` could be
every type of object. To overcome this we tell Cython to take a normal
C integer::

  def my_dot(x,y):

    if x.size != y.size
        raise ValueError("Dimension Mismatch")

    cdef double result = 0

    cdef int i

    for i in range(x.size):
        result += x[i]*y[i]

    return result
 
Now you can compile and use it. Let's measure the difference::

  sage: x = randn(10**6)
  sage: y = randn(10**6)
  sage: %timeit my_dot(x,y)
  5 loops, best of 3: 1.1 s per loop
  sage: load my_dot.spyx
  Compiling ./my_dot.spyx...
  sage: %timeit my_dot(x,y)
  5 loops, best of 3: 653 ms per loop

We this was already twice as fast as the old version. 
(I used a Pentium Dual Core with 1.8
GHz, and 2 GB Ram). This is not that much, but more is possible!

The next step would be to tell the function which data types to use::

  cimport numpy as cnumpy

  ctypedef cnumpy.float64_t reals #typedef_for easier reedding

  def my_dot(cnumpy.ndarray[reals,ndim=1] x,
                cnumpy.ndarray[reals,ndim=1] y):

      if x.size != y.size:
          raise ValueError("Dimension Mismatch")

      cdef double result = 0

      cdef int i

      for i in range(x.size):
          result += x[i]*y[i]

      return result

In the first line we used the ``cimport`` statement to load the C version
of NumPy. (I explain cimport later) 
Then we used the ``ctypedef`` statment to declare the float64 (double)
datatype as reals, so that we have to type less (like the typedef
statement in C).

The main difference in this example is that we told Cython that the
input should be to NumPy arrays. This avoids unecessary overhead. Now 
we make the timing again::

  sage: load my_dot.spyx
  Compiling ./my_dot.spyx...
  sage: %timeit my_dot(x,y)
  125 loops, best of 3: 3.54 ms per loop

This was now about 300x faster than the original version.

The drawback is that the Cython function only take numpy arrays::

  sage: x = range(5)
  sage: y = randn(5)
  sage: my_dot(x,y)
  ...
  TypeError: Argument 'x' has incorrect type (expected numpy.ndarray, got list)

Declaring functions
^^^^^^^^^^^^^^^^^^^

The ``cdef`` statement can also be used for defining functions. A
function that is defined by a ``cdef`` statment doesn't appear in the
namespace of the Python interpreter and can only be called within
other functions.

For example let's define a ``cdef`` function
``f``::

  cdef double f(double x):
      return x**2 - x

If yould try now to call it Python won't find it::

  NameError: name 'f' is not defined

But you can call it within an other function defined in a *.pyx* ::

  def call_f(double x):
      return f(x) 

Another possibility would be the cpdef statement::

    cdef double f(double x):
        return x**2 - x

This function can now be called both ways.

**Note:** If you don't declare it, ``cdef`` functions can't handly
exceptions right. For example
::

  cdef double f(double x):
    
      if x == 0:
          raise ValueError("Division by Zero!")

      return x**(-2) - x

would not raise a Python exception. To do this use the except
statement::

  cdef double f(double x) except *:
    
      if x == 0:
          raise ValueError("Division by Zero!")

      return x**(-2) - x

The ``*`` means that the function should propagate arbitrary 
exceptions. To be more specific you can also handle specific output::

  cdef double f(double x) except? 0:
    
    if x == 0:
        raise ValueError("Division by Zero!")

    return x**(-2) - x 

The ``?`` here means that ``0`` is accepted as output too (or else you
would recieve an error if ``0`` is returned)

``cdef`` classes
^^^^^^^^^^^^^^^^

Classes can also be defined with ``cdef`` also. Let's take the example
from the Cython documentation (see [#]_)::

  cdef class Function:
      cpdef double evaluate(self, double x) except *:
          return 0

A ``cdef`` class is also called Extension Type.

This class can be derived like a normal Python class::

  cdef class SinOfSquareFunction(Function):
      cpdef double evaluate(self, double x) except *:
          return sin(x**2)

``cdef`` classes are very limited in comparison to Python classes,
because C don't know classes, but only structs. (Since Cython 0.13 it
is possible to wrap C++ classes. See the Cython documentation for
further details [#]_) 

We can use this new class like a new datatype.
See again an example from the Cython documentation::

  def integrate(Function f, double a, double b, int N):
      cdef int i
      cdef double s, dx
      if f is None:
          raise ValueError("f cannot be None")
      s = 0
      dx = (b-a)/N
      for i in range(N):
          s += f.evaluate(a+i*dx)
      return s * dx

  print(integrate(SinOfSquareFunction(), 0, 1, 10000))


Calling extern C functions
""""""""""""""""""""""""""

In Cython it is possible to call functions from other C programs
defined in a header file. For example we want to wrap the sinus from
the math.h in a Python function. Then we would write for example in a
*.pyx* file::

  cdef extern from "math.h":
    double sin(double)

  def c_sin(double x):
      return sin(x)

The ``cdef extern`` statement help us to call ``sin`` from C. 
The ``c_sin`` function only serves as a wrapper for us, because we
can't call a ``cdef`` function directly. If you want to call your
Python function with sin, you can rename the extern C function with 
a custom made identifier::

  cdef extern from "math.h":
      double c_sin "sin"(double)

  def sin(double x):
      return c_sin(x) 

The ``c_sin`` is the name of the ``cdef`` function.

If you want to compile this file, you have to tell your compiler which
libraries you linked, because they are not linked automatically! In
this case it is the math library with abbreviation "m". You have to
specify this in your setup file (I saved the sinus to *math_stuff.pyx*)::

  from distutils.core import setup
  from distutils.extension import Extension
  from Cython.Distutils import build_ext

  setup(
      name = "Math Stuff",
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("math_stuff", ["math_stuff.pyx"],
                     libraries=["m"])]
      #m for compiler flag -lm (math library)
  ) 

If you use Sage you have to specify this directly in the *.spyx* file
with::

  #clib m

in our example this would look like this::

  #clib m 

  cdef extern from "math.h":
      double c_sin "sin"(double)

  def sin(double x):
      return c_sin(x)


Another example: Let's link the scalar product from the BLAS
library::

  cimport numpy

  ctypedef numpy.float64_t reals #typedef_for easier reedding

  cdef extern from "cblas.h":
      double ddot "cblas_ddot"(int N, double *X, int incX,double *Y, int incY)
    
  def blas_dot(numpy.ndarray[reals,ndim = 1] x, numpy.ndarray[reals,ndim = 1] y):
      return ddot(x.shape[0],<reals*>x.data,x.strides[0] // sizeof(reals), <reals*>y.data,y.strides[0] // sizeof(reals))

The blas implementation gives only a small improvement here (which is
not completely unexpected, because the algorithm is rather simple)::

  sage: x = randn(10**6)
  sage: y = randn(10**6)
  sage: %timeit my_dot(x,y)
  125 loops, best of 3: 3.55 ms per loop
  sage: %timeit blas_dot(x,y)
  125 loops, best of 3: 3.05 ms per loop

``cimport`` and .pxd files
""""""""""""""""""""""""""

*.pxd* are like *.h* files in C. They can be used for sharing external
C declarations, or functions that are suited for inlining by the C compiler.

Functions that are declared inline in *.pxd* files can be imported with the
``cimport`` statement.

For example let's add a function which calculates the square root of
a number to the *math_stuff.pyx* from earlier, where the operation
itelf is called as inline function from C. We write the inline
function to the file *math_stuff.pxd*::

  cdef inline double inl_sqrt(double x):
      return x**(0.5)
  
We can now load this function from a *.pyx* file::

  def sqrt(double x):
      return inl_sqrt(x)  

You can also save the extern definition of the BLAS scalar product to
a *.pxd* file and can ``cimport`` it from there.

Here the *blas.pxd* file::

  cdef extern from "cblas.h":
      double ddot "cblas_ddot"(int N, double *X, int incX,double *Y, int incY)

and here the addition to the math_stuff.pyx::

  cimport numpy

  from blas cimport ddot
  ctypedef numpy.float64_t reals #typedef_for easier reedding
  
  cpdef dot(numpy.ndarray[reals,ndim = 1] x, numpy.ndarray[reals,ndim = 1] y):
      return ddot(x.shape[0],<reals*>x.data,x.strides[0] //
      sizeof(reals), <reals*>y.data,y.strides[0] // sizeof(reals))


What is also possible is to declare prototypes in a *.pxd* file like
in a C header, which can be linked more efficiently.

For example let's make a prototype of a function and a class::

  cpdef dot(numpy.ndarray[reals,ndim = 1] x, numpy.ndarray[reals,ndim = 1] y)

  cdef class Function:
      cpdef double evaluate(self, double x)

Profiling
"""""""""

Profiling is a way to analyse and optimize your Cython programs.
I only give the reference to a tutorial in the Cython documentation
here  [#]_

.. rubric:: Links

.. [#] http://docs.cython.org/index.html
.. [#] http://docs.cython.org/src/tutorial/cdef_classes.html
.. [#] http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
.. [#] http://docs.cython.org/src/tutorial/profiling_tutorial.html
