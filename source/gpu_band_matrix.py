# -*- coding: utf-8 -*-
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from pycuda.driver import matrix_to_texref

from numpy import array, zeros, int32, float32, intp

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
          result_helper += tex2D(matrixTexture, i, idx - i)*vector_help[threadIdx.x + %(BLOCK_SIZE)s - i];
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
    matvec_func(intp(x_gpu.gpudata),intp(y_gpu.gpudata), self.gpu_info.gpudata, block = (BLOCK_SIZE,1,1), grid= (NUMBER_BLOCKS,1))
    return y_gpu
    
  def __add__(self,other):
    return add_band_mat(self,other)
