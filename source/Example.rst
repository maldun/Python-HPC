An Example: Band-matrix vector multiplication
=============================================

We want to implement a band-matrix class for a symmetric band-matrix.
The constructor  takes an array as input, which holds the matrix
entries, of the lower part of the band matrix. See Wikipedia for
an Idea [#]_, and the IBM ESSL for precise details [#]_.

First we implement our class with several class methods, like
addition, and a matvec method::

  def add_band_mat(A,M,beta = 1,alpha = 1):
      (d1,d2) = A.shape
      (d3,d4) = M.shape
      
      d1 = A.band_width
      d3 = M.band_width
      
      if d2 != d4:
          raise ValueError("From _rational_krylov_trigo_band:\
                            Dimension Missmatch!")

      if (d1 < d3):
          SYST = py_band_matrix(zeros((d3,d4)));
          SYST.data[0:d1,0:d2] = beta*A.data;
          SYST.data = alpha*M.data + SYST.data;
      elif (d1 > d3):
          SYST = py_band_matrix(zeros((d1,d2)));
          SYST.data[0:d3,0:d4] = alpha*M.data;
          SYST.data = SYST.data + beta*A.data;
      else:
          SYST =  py_band_matrix(alpha*M.data + beta*A.data);
    
      return SYST

  class band_matrix:
      def __init__(self,ab):
          self.shape = (ab.shape[1],ab.shape[1])
          self.data = ab
          self.band_width = ab.shape[0]
          self.dtype = ab.dtype
      
      def matvec(self,u):
          pass
      
      def __add__(self,other):
          return add_band_mat(self,other)

First we implement our matrix vector multiplication in Python::

  def band_matvec_py(A,u):

      result = zeros(u.shape[0],dtype=u.dtype)

    
      for i in xrange(A.shape[1]):
          result[i] = A[0,i]*u[i]

      for j in xrange(1,A.shape[0]):
          for i in xrange(A.shape[1]-j):
              result[i] += A[j,i]*u[i+j]
              result[i+j] += A[j,i]*u[i]

      return result

Then we derive our Python base class with the Python matrix vector
multiplication::

  class py_band_matrix(band_matrix):
      def matvec(self,u):
          if self.shape[0] != u.shape[0]:
              raise ValueError("Dimension Missmatch!")
          
          return band_matvec_py(self.data,u)


But this is quite slow. We can alternativly implement this Inline with weave::

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

and create a new band matrix class::

  class inline_band_matrix(band_matrix):
      def matvec(self,u):
          if self.shape[0] != u.shape[0]:
              raise ValueError("Dimension Missmatch!")
          
          return band_matvec_inline(self.data,u)

or we implement the matrix vector product with Cython::

  cimport numpy as cnumpy
  ctypedef cnumpy.float64_t reals #typedef_for easier reedding

  def band_matvec_c(cnumpy.ndarray[reals,ndim=2] A,cnumpy.ndarray[reals,ndim=1] u):
      cdef Py_ssize_t i,j
      cdef cnumpy.ndarray[reals,ndim=1] result = numpy.zeros(A.shape[1],dtype=A.dtype) 
      for i in xrange(A.shape[1]):           
          result[i] = A[0,i]*u[i]
      
      for j in xrange(1,A.shape[0]):
          for i in xrange(A.shape[1]-j):
              result[i] = result[i] + A[j,i]*u[i+j]
              result[i+j] = result[i+j]+A[j,i]*u[i] 

      return result

and make the new band-matrix class analogously::

   class c_band_matrix(band_matrix):
      def matvec(self,u):
          if self.shape[0] != u.shape[0]:
              raise ValueError("Dimension Missmatch!")
          
          return band_matvec_c(self.data,u)

You can either import the band matrix base class to the *.pyx* file
and define the derived Python class in the *.pyx* file, or ``cimport``
the function to a Python file with the new matrix class defined



.. rubric:: Links

.. [#] http://en.wikipedia.org/wiki/Band_matrix 
.. [#] http://publib.boulder.ibm.com/infocenter/clresctr/vxrx/index.jsp?topic=%2Fcom.ibm.cluster.essl43.guideref.doc%2Fam501_upbsm.html


