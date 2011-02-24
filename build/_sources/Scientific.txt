Scientific tools in Python
==========================

NumPy
-----

NumPy is based on the old Python Project Numeric, and introduces a
Matlab like vector class to Python. Numpy is currently developed by
Entought and is distrubuted und the BSD license. Further Information 
and intall instructions can be found on the official NumPy website [#]_.
Sage and FemHUB are shipped with current versions of NumPy.

The project is still in an active development process, and release
new versions in an regular basis (The last release before I started to
writing this report is 2 Months old)

For people who are familiar with Matlab I recommend the
online equivalence list between Matlab and Numpy from *Mathesaurus* [#]_
for the first steps with NumPy.

How to load Numpy
"""""""""""""""""

To import numpy into Python simply write::

  import numpy

You can also import the whole namespace via ``*``::

  from numpy import *

**Remark** I personally don't recommend to load the complete
nammespace, or the complete package, except for testing, due to
performance reasons. (This is somehow obvious because NumPy is
a rather big module)

Numpy arrays
""""""""""""
The NumPy *array* class is rather similar to Python *lists*.
It's the basic datatype for many numerical tools in Python.

Array Creation
^^^^^^^^^^^^^^

To create an *array* we have to import it from NumPy first::

  from numpy import array 

An *array* can be created from different Python sequence types like
*lists* or tuples. For example::

  >>> l = [1,2,3]
  >>> t = (1,2,3)
  >>> array(l)
  array([1, 2, 3]) 
  >>> array(t)
  array([1, 2, 3])

**Remark** It is not clerly specified by the documention which other
containers may work (I guess the reason for this is that the
constructor is written in a quite generic way. The Python way to find
out if it work with other tzpes would be testing it out,

The intention of the NumPy developers was to give a Matlab like
feeling. So Many ways should be quite familiar for Matlab users::

  >>> from numpy import zeros, ones, eye
  >>> from numpy.random import rand
  >>> zeros(3)
  array([ 0.,  0.,  0.])
  >>> ones(4)
  array([ 1.,  1.,  1.,  1.])
  >>> eye(2)
  array([[ 1.,  0.],
        [ 0.,  1.]])
  >>> rand(4)
  array([ 0.62475625,  0.97783392,  0.7785848 ,  0.15707817])
  >>> zeros((2,2))      #create matrix
  array([[ 0.,  0.],
         [ 0.,  0.]])

A NumPy array can hold numbers of specific data types. To check which
datatype an array holds, one has to simply check the ``dtype``
member::

  >>> l = array([1,2])
  >>> l.dtype
  dtype('int64')
  >>> l2 = array([1.,2])
  >>> l2.dtype
  dtype('float64')

As one can see, the default datatpe for integers is ``int64``,
while for floating point numbers it is ``float64`` (because it is
a 64 bit system I am working on). But there are some more::

  from numpy import float32  #single precision
  from numpy import float64  #double precision
  from numpy import float128 #long double

  from numpy import int16  #16 Bit integer
  from numpy import int32  #32 Bit integer 
  from numpy import int64  #64 Bit integer 
  from numpy import int128 #128 Bit Integer 

To create an array with a specific data type, you only have to 
specify this::

  >>> array([2,3],int32)     
  array([2, 3], dtype=int32)
  >>> array([2,3],dtype=int32)  #using keyword argument
  array([2, 3], dtype=int32)

But these aren't all possible datatypes. NumPy support also other
types, and the number is still growing, since it is under development. 

There are several other ways to create arrays. See
the NumPy documentation [#]_, [#]_ for further details.

Artithmetics with NumPy arrays
""""""""""""""""""""""""""""""
Since operators can be overloaded (see ::ref::`overload_ref` , NumPy supports also arithmetics
with *arrays*. Note that all operations are elementwise.
::

  >>> a = array([2.,3]); b = array([5.,6])   # Create vectors 
  >>> a + b                                  
  array([ 7.,  9.])
  >>> a - b                                  
  array([-3., -3.])
  >>> a * b
  array([ 10.,  18.])
  >>> a / b
  array([ 0.4,  0.5])
  >>> a ** b
  array([  32.,  729.])

To calculate the scalar product one has to use the ``dot`` function::

  >>> from numpy import dot
  >>> dot(a,b)
  28.0

With the help of dot you can also calculate the matrix vector
product::

  >>> A = ones((2,2))
  >>> A
  array([[ 1.,  1.],
         [ 1.,  1.]])
  >>> dot(A,a)
  array([ 5.,  5.])

Applying functions elementwise
""""""""""""""""""""""""""""""

NumPy also holds a lot of standard functions for elementwise
operations::

  >>> from numpy import sin, cos
  >>> sin(a)
  array([ 0.90929743,  0.14112001])
  >>> cos(a)
  array([-0.41614684, -0.9899925 ])

(see the NumPy reference guide for further information [#]_)

To create your own customized elementwise functions use the ``vectorize`` class in
NumPy. It takes a Python function for construction of the object, and
vectorize it.

Examples::

  from numpy import vectrorize, array
  from numpy.random import randn

  def my_sign(x):    
      if  x > 0:
          return 1
      elif x < 0:
          return -1
      else:
          return 0

  vec_abs = vectorize(my_sign)

Then we get::

  >>> vec = randn(10); vec 
  array([ 1.2577085 ,  0.71063021,  1.41130699,  1.72412141, -1.18530781,
          0.19527091, -0.20557102, -0.33562998, -1.5370958 , -0.47241905])
  >>> vec_abs(vec)
  array([ 1,  1,  1,  1, -1,  1, -1, -1, -1, -1])
  
SciPy
-----

SciPy is a module for scientific computing. It is based on NumPy and
holds a lot of extensions and algorithms. In fact NumPy is subsumed
in SciPy already.
It contains a lot of functionality which is contained in Matlab.

I will explain some scientific tools in detail, which are of common
interest.

Linear Algebra
""""""""""""""

For doing linear algebra with SciPy I would prefer to point at the
SciPy documentation, because it is much more detailed [#]_ 

Sparse Linear Algebra
"""""""""""""""""""""

There are several types of sparse matrices. Each of them has several
attributes and is used for different tasks. 
I introduce here the ones I use the most, and some other important
features like the LinearOperator class.

LIL (List of Lists)
^^^^^^^^^^^^^^^^^^^
LIL matrices are made for creating sparse matrices.
To create a LIL matrix simply import the class and call the
constructor::

  >>> from scipy.sparse import lil_matrix
  >>> A = lil_matrix((1000,1000))

Now we can fill the entries like we do it normally with numpy vectors::

  >>> from scipy import rand
  >>> A[0, 0:100] = rand(100); A
  <1000x1000 sparse matrix of type '<type 'numpy.float64'>'
	with 100 stored elements in LInked List format>
  >>> A[1:21,1:21] = rand(20,20); A
  <1000x1000 sparse matrix of type '<type 'numpy.float64'>'
	with 500 stored elements in LInked List format>

  and of course call the entries directly::

  >>> A[1,1]
  0.85312312525865719

LIL matrices are not suited for arithmetics or vector operations but
for creating other sparse matrices. To convert it into an other sparse
type simply call the converting methods. Lets convert it for example
to CSC ( Compressed Sparse Column matrix) format::

  >>> A_csc = A.tocsc()
  >>> A_csc
  <1000x1000 sparse matrix of type '<type 'numpy.float64'>'
  	with 500 stored elements in Compressed Sparse Column format>

To convert it back to a numpy vector simply call::

  >>> A.toarray()
  array([[ 0.16568301,  0.85841039,  0.58243887, ...,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.85312313,  0.33507849, ...,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.97454761,  0.16457123, ...,  0.        ,
           0.        ,  0.        ],
         ..., 
         [ 0.        ,  0.        ,  0.        , ...,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        , ...,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        , ...,  0.        ,
           0.        ,  0.        ]])

CSC (Compressed Sparse Column) matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CSC matrices are quite often used because they can perform matrix
vector multiplication quite efficiently. To create a CSC matrix either
do it with a LIL matrix like in the LIL matrix section before, or
create it with three arrays which contain the necessary data::

  >>> from scipy.sparse import csc_matrix
  >>> from numpy import array
  >>> rows = array([0,2,2,1,0])
  >>> cols = array([0,2,0,2,1])
  >>> data = array([1,2,3,4,5])
  >>> B = csc_matrix((data, (rows,cols)), shape = (3,3)); B.toarray()
  array([[1, 5, 0],
         [0, 0, 4],
         [3, 0, 2]])

Another variant would be the standard CSC representation. There are
three arrays: an index_pointer array, an indices array, and a data
array. The row indices  for the ::math::`i` th row are stored in 
``indices[index_pointer[i],index_pointer[i+1]]``, while their 
corresponding data is stored in
``data[index_pointer[i]:index_pointer[i+1]]``.
So the ``index_pointer`` tells where to start and to stop
while going throug the indices and data lists. For example::

  >>> indptr = array([0,2,3,6])
  >>> indices = array([0,2,2,0,1,2])
  >>> data = array([1,2,3,4,5,6])
  >>> csc_matrix( (data,indices,indptr), shape=(3,3) ).toarray()
  array([[1, 0, 4],
         [0, 0, 5],
         [2, 3, 6]])
  
Other possible ways would be generating the matrix with another
sparse matrix or an dense 2D array with the data as constructing
data::

  >>> csc_matrix(array([[0,1],[1,0]]))
  <2x2 sparse matrix of type '<type 'numpy.int32'>'
	  with 2 stored elements in Compressed Sparse Column format>
  >>> csc_matrix(array([[0,1],[1,0]])).toarray()
  array([[0, 1],
         [1, 0]])
   
...and more
^^^^^^^^^^^

To get more information on sparse matrices and their class methods
consult the scipy reference guide [#]_

The LinearOperator class and iterative solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LinearOperator class allows to define abstract linear mappings,
which are not necessarily matrices. A linear operator only consists of
a tuple which represents the shape, and a matrix-vector
multiplication::

  >>> def my_matvec(x):
  ...     a = x[-1]
  ...     x[-1] = x[0]
  ...     x[0] = a
  ...     return x
  ... 
  >>> from scipy.sparse.linalg import LinearOperator
  >>> lin = LinearOperator((3,3),matvec=my_matvec)
  >>> x = array([1,2,3])
  >>> lin.matvec(x)
  array([3, 2, 1])
  
The matrix vector multiplication can also be called with the `*` operator::

  >>> lin*x
  array([1, 2, 3])
  >>> lin * x
  array([3, 2, 1])

LinearOperators can be created from arrays, matrices or sparse
matrices with the `aslinearoperator` function::

  >>> A = array([[2,-1,0],[-1,2,-1],[0,-1,2]])
  >>> from scipy.sparse.linalg import aslinearoperator
  >>> A_lin = aslinearoperator(A)
  >>> A_lin.matvec(x)
  array([4, 0, 0])


The LinearOperator class is mostly used for iterative Kylov solvers. Those
methods can be found in the `scipy.sparse.linalg`. For example the
CG algorithm::

  >>> from scipy.sparse.linalg import cg
  >>> A_lin*sol[0]
  array([ 3.,  2.,  1.])
  >>> x
  array([3, 2, 1])  

For more information see again the SciPy reference [#]_

.. _weave_ref:

Weave
-----

Weave is included in SciPy and a tool for writing inline C++ with
weave for speedup your code. I give here a short example how to use
Weave.

Consider band-matrix vector multiplication::

  def band_matvec_py(A,u):

      result = zeros(u.shape[0],dtype=u.dtype)

    
      for i in xrange(A.shape[1]):
          result[i] = A[0,i]*u[i]

      for j in xrange(1,A.shape[0]):
          for i in xrange(A.shape[1]-j):
              result[i] += A[j,i]*u[i+j]
              result[i+j] += A[j,i]*u[i]

      return result

This is not very fast::

  sage: import numpy
  sage: datatype = numpy.float64
  sage: N = 2**14
  sage: B = 2**6
  sage: A = rand(B,N).astype(datatype)
  sage: %timeit band_matvec_py(A,u)
  5 loops, best of 3: 3.48 s per loop

The reason for this is that array access is quite costly
in Python. A possibility to make that better would be to write 
C++ code inline with the Weave module. To do that give the Python
Interpreter your C++ code as string, and then let compile it. Here
an implementation of the band-matrix vector multiplication with weave::

  from numpy import array, zeros
  from scipy.weave import converters
  from scipy import weave

  def band_matvec_inline(A,u):

      result = zeros(u.shape[0],dtype=u.dtype)

      N = A.shape[1]
      B = A.shape[0]
    
      code = """
      for(int i=0; i < N;i++)
      {
        result(i) = A(0,i)*u(i);
      }
      for(int j=1;j < B;j++)
      {
      
          for(int i=0; i < (N-j);i++)
          {
            if((i+j < N))
            {
              result(i) += A(j,i)*u(j+i);
              result(i+j) += A(j,i)*u(i);
            }
         
          }  
         
      }
      """

      weave.inline(code,['u', 'A', 'result', 'N', 'B'],
                 type_converters=converters.blitz)
      return result

As it can be seen the syntax is not that different from Numpy. The 
reason for this is, that Weave uses here the Blitz library for
numerical computation, which has it's own vector class.

If you call this function the first time it will be compiled in
runtime::

  sage: band_matvec_inline(A,u)
  creating /tmp/maldun/python26_intermediate/compiler_2da6387b1d12110fba46fe47fea9326a
  In file included from /home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/scipy/weave/blitz/blitz/array-impl.h:37,
                   from /home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/scipy/weave/blitz/blitz/array.h:26,
                   from /home/maldun/.python26_compiled/sc_7f8ca882b38e1f398003844545921f4a0.cpp:11:
  /home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/scipy/weave/blitz/blitz/range.h: In member function ‘bool blitz::Range::isAscendingContiguous() const’:
  /home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/scipy/weave/blitz/blitz/range.h:120: warning: suggest parentheses around ‘&&’ within ‘||’
  array([-7.03708979, -0.53476595, -7.52383126, ...,  1.18391403,
          2.27257052,  0.39116477])

The next time you call it, the interpreter will use the compiled
program.
Let's test the speedup::

  sage: %timeit band_matvec_inline(A,u)
  25 loops, best of 3: 12.7 ms per loop

This was now about 270x faster than the original Python version.
For more information on using weave see either the documentation of
SciPy [#]_ or the Sage tutorial on that topic [#]_.

**Note:** At the time I checked the Sage tutorial the last time
it was not updated and contain some mistakes. In the next version of
Sage (4.6.2) this should be corrected. See the Sage trac for a corrected
version [#]_  

.. rubric:: Links

.. [#] http://numpy.scipy.org/
.. [#] http://mathesaurus.sourceforge.net/matlab-numpy.html 
.. [#] http://docs.scipy.org/doc/numpy-1.5.x/user/basics.creation.html#arrays-creation
.. [#] http://docs.scipy.org/doc/numpy-1.5.x/reference/routines.array-creation.html#routines-array-creation
.. [#] http://docs.scipy.org/doc/numpy/reference/routines.math.html
.. [#] http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
.. [#] http://docs.scipy.org/doc/scipy/reference/sparse.html
.. [#] http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
.. [#] http://www.scipy.org/Weave
.. [#] http://www.sagemath.org/doc/numerical_sage/weave.html
.. [#] http://trac.sagemath.org/sage_trac/ticket/9791

        



