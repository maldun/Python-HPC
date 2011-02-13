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
NumPy. It takes 

  

.. rubric:: Links

.. [#] http://numpy.scipy.org/
.. [#] http://mathesaurus.sourceforge.net/matlab-numpy.html 
.. [#] http://docs.scipy.org/doc/numpy-1.5.x/user/basics.creation.html#arrays-creation
.. [#] http://docs.scipy.org/doc/numpy-1.5.x/reference/routines.array-creation.html#routines-array-creation
.. [#] http://docs.scipy.org/doc/numpy/reference/routines.math.html



        



