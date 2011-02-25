An Example: Band-matrix vector multiplication
=============================================

We want to implement a band-matrix class for a symmetric band-matrix.
The constructor  takes an array as input, which holds the matrix
entries, of the lower part of the band matrix. See Wikipedia for
an Idea [#]_, and the IBM ESSL for precise details [#]_.

First we implement our class with several class methods, like
addition and a matvec method::

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
the function to a Python file with the new matrix class defined.

.. highlight:: c

For a comparison, the algorithm implemented in C::

  
  void band_matvec(float *result, float *A, float *x)
  {
    unsigned int i,j;
    for(i = 0; i < cols; i++)
    {
      result[i] = A[i]*x[i]; 
    }
    
    for(j = 1; j < rows; j++)
    {
      for(i = 0; i < cols - j; i++)
      {
          result[i] += A[j + i*rows]*x[i+j];
          result[i+j] += A[j + i*rows]*x[i];
      	
      }
    }
  }

.. highlight:: matlab

and a (direct) Matlab version::

  function result = band_matvec_ma(A,u)
  
      [B N] = size(A);
    
      result = zeros(N,1);
    
      
      for i = 1:N
          result(i) = A(1,i)*u(i);
      end
      
      for j = 2:B
          for i =1:N
              if ((i+j-1) <= N) 
                result(i) = result(i) + A(j,i)*u(i+j-1);
                result(i+j-1) = result(i+j-1) + A(j,i)*u(i);
              end
          end
      end
  end

I made a time comparison on a Intel Core Duo with 1800MHz and
2 GB Ram. The dimension was ::math::``2^14``, and a with a 
band of the size of ::math::``2^6`` under the diagonal. The reason
for this choice was the limited texture memory on the card to
compare it with the PyCUDA implementation lateron.

============================= ======= =======
Language                       time   Speedup
============================= ======= =======
Python                         2.96 s    1x
Matlab                         90 ms    30x
C                              30 ms    60x
Cython                         12 ms   220x
Inline C++                     10 ms   290x
C (-O2 compiler flag)          10 ms   290x
Cython (w. Boundscheck false)   6 ms   440x
============================= ======= =======

**Remarks**: 

* @Matlab: the for loops are bad rule is no longer true! The
  Java JIT (Just in Time) compilation vectorizes simple for loops like 
  mine very efficiently, and that's also the reason why the Matlab
  version is 30 times faster than the Python implementation. There
  is a JIT compiler for Python too named Psyco. But Psyco doesn't
  optimize the numpy array access like Cython.

* @C The C implementation is rather straight forward and not well
  optimised. Of course C cannot be slower than Cython, because
  Cython is compiled to C. 

There is a saying: "There are lies,
there are big lies, and there are benchmarks", and my intention
here is not to show that Python/Cython is the fastest language,
but it is very easy to write fast Code in that languages, which can
be faster than other implementations, if you don't optimize them well.
That is a big Plus
for Python/Cython, because when you write a prototype you want to get
your 
code fast enough that you can work with it, and don't have to invest
much time with optimizing it. Although well written Cython code is 
nearly as fast as a direct C implementation. If you don't need the
last bit of speed out of your programs it is a good and simple
alternative. 

.. highlight:: python

A more advanced example is an implementation in PyCUDA::

  import pycuda.driver as cuda
  import pycuda.gpuarray as gpuarray
  
  import pycuda.autoinit
  import numpy
  from pycuda.compiler import SourceModule
  from pycuda.driver import matrix_to_texref
  
  from numpy import array, zeros, int32, float32, intp
  
.. highlight:: c

::

  sourceCodeTemplate = """
  texture<float, 2> matrixTexture;
  
  __global__ void gpu_band_matvec( //%(REALS)s *matrix, 
                                   %(REALS)s *vector,
                                   %(REALS)s *result,
                                   int *info
                                   ) 
  {
    // Infos about matrix dimension
    const int dim = info[0];
    const int band_with = info[1];
      
    // Infos about Blocks

    //unsigned int position = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
  
    __shared__ %(REALS)s vector_help[%(DOUBLE_BLOCK_SIZE)s];
    %(REALS)s result_helper;
  
    //__syncthreads(); //Memory has to be loaded
  
    while(idx < dim +  %(BLOCK_SIZE)s * %(NUMBER_BLOCKS)s) //While loop over all blocks
    {
    
        __syncthreads(); //Memory has to be loaded
        
        if(idx < dim)
        {
        vector_help[threadIdx.x] = vector[idx]; 
        
        
        if(idx + %(BLOCK_SIZE)s < dim)
        {
          vector_help[threadIdx.x+%(BLOCK_SIZE)s] =  vector[idx+%(BLOCK_SIZE)s]; 
        }
        __syncthreads(); //Memory has to be loaded
        
        result_helper = tex2D(matrixTexture, 0, idx) * vector_help[threadIdx.x];
      
        for(int i = 1; i < band_with; i++)
        {
           result_helper += tex2D(matrixTexture, i, idx)*vector_help[threadIdx.x+i];
         
        } 
        
        
       __syncthreads(); //Memory has to be loaded
       
       vector_help[threadIdx.x + %(BLOCK_SIZE)s] = vector_help[threadIdx.x];
       }
       
       if((idx - %(BLOCK_SIZE)s >= 0) && (idx - %(BLOCK_SIZE)s) < dim)
       {
         vector_help[threadIdx.x] = vector[idx - %(BLOCK_SIZE)s];
       }
       
       __syncthreads(); //Memory has to be loaded
       
       for(int i = 1; i < band_with; i++)
       {
          if((idx - i >= 0) && (idx - i < dim))
          {
            result_helper += tex2D(matrixTexture, i, idx - i)*
            vector_help[threadIdx.x + %(BLOCK_SIZE)s - i];
          }
       }
      
       if(idx < dim)
       {
         result[idx] = result_helper;
       }
      
      
      idx += %(BLOCK_SIZE)s * %(NUMBER_BLOCKS)s;
      
     }
  }
  
  """

.. highlight:: python

::

  REALS = "float"
  BLOCK_SIZE = 256
  NUMBER_BLOCKS = 8
  
  sourceCode = sourceCodeTemplate % {
  'REALS': REALS, 
  'BLOCK_SIZE': BLOCK_SIZE,
  'DOUBLE_BLOCK_SIZE': 2*BLOCK_SIZE,
  'NUMBER_BLOCKS': NUMBER_BLOCKS,
  }
  
  matvec_module = SourceModule(sourceCode)
  
  matvec_func = matvec_module.get_function("gpu_band_matvec")
  matrixTexture = matvec_module.get_texref("matrixTexture")
  
  from band_matrix import add_band_mat
  
  class gpu_band_matrix:
    """variables for information about which matrix is
       currently on the texture
    """
  
    nr_matrices = 0
    active_matrix = 0
  
    """Takes a numpy array"""
    def __init__(self,data,set_right = False):
        self.data = data
  
        if set_right:
            for j in xrange(1,B):
                self.data[j:,N-j] = 0
    
        #self.data = gpuarray.to_gpu(data)
        self.shape = (data.shape[1],data.shape[1])
        self.band_with = data.shape[0]
        self.dtype = data.dtype
        
        info = array([data.shape[1],data.shape[0]]).astype(int32)
        
        self.gpu_info = gpuarray.to_gpu(info)
        
        cuda.matrix_to_texref(data, matrixTexture, order="F")
        
        gpu_band_matrix.nr_matrices += 1
        
        self.identity_nr = gpu_band_matrix.nr_matrices
        active_matrix = gpu_band_matrix.nr_matrices

    def matvec(self,x_gpu):
        if x_gpu.size != self.shape[0]:
            raise ValueError("Dimension mismatch!")
    
        #self.data.bind_to_texref_ext(matrixTexture,channels = 2)
        #cuda.matrix_to_texref(self.cpu_data, matrixTexture, order="F")
        
        if gpu_band_matrix.active_matrix != self.identity_nr:
            cuda.matrix_to_texref(self.data, matrixTexture, order="F")
            gpu_band_matrix.active_matrix = self.identity_nr
        
        y_gpu  = gpuarray.empty(x_gpu.size, x_gpu.dtype)
        matvec_func(intp(x_gpu.gpudata),intp(y_gpu.gpudata),
        self.gpu_info.gpudata, block = (BLOCK_SIZE,1,1), grid= (NUMBER_BLOCKS,1))
        return y_gpu
    
    def __add__(self,other):
        return add_band_mat(self,other)

I tested it on the GPU4U at the University of Graz and had the
following times:

============================= ======= =======
Language                       time   Speedup
============================= ======= =======
Python                        2.87 s    1x
Cython                        9.87 ms   270x
Cython (w. Boundscheck false) 5.56 ms   500x
PyCUDA (on GTX 280)            585 µs  5000x
============================= ======= =======

**Remark**: PyCUDA is very fast, but has a breakeven point!
For small dimensions Python or Cython are much faster.

.. rubric:: Links

.. [#] http://en.wikipedia.org/wiki/Band_matrix 
.. [#] http://publib.boulder.ibm.com/infocenter/clresctr/vxrx/index.jsp?topic=%2Fcom.ibm.cluster.essl43.guideref.doc%2Fam501_upbsm.html


