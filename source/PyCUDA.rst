Python+CUDA = PyCUDA
====================

PyCUDA is a Python Interface for CUDA [#]_. It is currently in Alpha
Version, and was developed by Andreas Klöckner [#]_

To use PyCUDA you have to install CUDA on your machine

**Note:** For using PyCUDA in Sage or FEMHub I created a PyCUDA
  package [#]_.

I will give here a short introduction how to use it. For more detailed
Information I refer to the documentation [#]_ or the Wiki [#]_.

Initialize PyCUDA
-----------------

There are two ways to initialize the PyCUDA driver. The first one is
to use the autoinit module::

import pycuda.autoinit

This makes the first device ready for use. Another possibility
is to manually initialize the device and create a context on this
device to use it::

  import pycuda.driver as cuda
  cuda.init() #init pycuda driver
  current_dev = cuda.Device(device_nr) #device we are working on
  ctx = current_dev.make_context() #make a working context
  ctx.push() #let context make the lead

  #Code

  ctx.pop() #deactivate again
  ctx.detach() #delete it
  
This is useful if you are working on different devices. I will give 
a more detailed example combined with MPI4Py lateron. 
(See ::ref::`mpi_and_pycuda_ref`)

Get your CUDA code working in Python
------------------------------------

Similar to ::ref::`weave_ref` we can write CUDA code as string in
Python and then compile it with the NVCC. Here a short example:

First we initialize the driver, and import the needed modules::

  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy
  from pycuda.compiler import SourceModule
 
Then we write our Source code::

  code = """
  __global__ void double_array_new(float *b, float *a, int *info)
  {
    int datalen = info[0];
  
    for(int idx = threadIdx.x; idx < datalen; idx += blockDim.x)
    {
      b[idx] = a[idx]*2;
    }
  }
  """

And then write it to a source module::

  mod = SourceModule(code)

The NVCC will now compile this code snippet. Now we can load the new
function to the Python namespace::

  func = mod.get_function("double_array_new")

Let's create some arrays for the functions, and load them on the card::

  N = 128

  a = numpy.array(range(N)).astype(numpy.float32)
  info = numpy.array([N]).astype(numpy.int32)
  b = numpy.zeros_like(a)

  a_gpu = cuda.mem_alloc(a.nbytes)
  cuda.memcpy_htod(a_gpu, a)

  b_gpu = cuda.mem_alloc(b.nbytes)
  cuda.memcpy_htod(b_gpu, b)

  info_gpu = cuda.mem_alloc(info.nbytes)
  cuda.memcpy_htod(info_gpu, info)

Now we can call the function::

  func(b_gpu, a_gpu,info_gpu, block = (32,1,1), grid = (4,1))

**Note:** The keyword ``grid`` is optional. If no grid is assigned,
it consists only of one block.

Now get the data back to the host, and print it::

  a_doubled = numpy.empty_like(a)
  cuda.memcpy_dtoh(a_doubled, b_gpu)

  print "result:", a_doubled

**Note:** To free the memory on the card use the free method::

  a_gpu.free()
  b_gpu.free()
  info_gpu.free()

PyCUDA has Garbage Collection, but it's still under developement. I
Therefore recommend it to free data after usage, just to be sure.

To create a Texture reference, to bind data to a texture on the
Graphic card. you have first to create one your source code::

  code_snippet = """
  texture<float, 2> MyTexture;
  // Rest of Code
  """

Then compile it::

  >>> texture_mode = SourceModule(code_snippet)

and get it::

  >>> MyTexture = texture_mode.get_texref("MyTexture")


The ``gpuarray`` class
----------------------

The ``gpuarray`` class provides a high level interface for doing
calculations with CUDA.
First import the gpuarray class::

  >>> import pycuda.driver as cuda
  >>> import pycuda.autoinit
  >>> from pycuda import gpuarray

Creation of gpuarrays is quite easy. One way is to create a NumPy
array and convert it::

  >>> from numpy.random import randn
  >>> from numpy import float32, int32, array
  >>> x = randn(5).astype(float32)
  >>> x_gpu = gpuarray.to_gpu(x)

You can print gpuarrays like you normally do::

  >>> x
  array([-0.24655211,  0.00344609,  1.45805557,  0.22002029,  1.28438667])
  >>> x_gpu
  array([-0.24655211,  0.00344609,  1.45805557,  0.22002029,  1.28438667]) 


You can do normal calculations with the gpuarray::

  >>> 2*x_gpu
  array([-1.09917879,  0.56061697, -0.19573164, -4.29430866, -2.519032  ], dtype=float32)  

  >>> x_gpu + x_gpu
  array([-1.09917879,  0.56061697, -0.19573164, -4.29430866, -2.519032  ], dtype=float32)

or check attributes like with normal arrays::

  >>> len(x_gpu)
  5

``gpuarrays`` also support slicing::

  >>> x_gpu[0:3]
  array([-0.5495894 ,  0.28030849, -0.09786582], dtype=float32)

Unfortunatly they don't support indexing (yet)::

  >>> x_gpu[1]
  ...
  ValueError: non-slice indexing not supported: 1

Be aware that a function which was created with a SourceModule, takes
an instance of ``pycuda.driver.DeviceAllocation`` and not a gpuarray.
But the content of the ``gpuarray`` is a ``DeviceAllocation``. You can
get it with the attribute ``gpudata``::

  >>> x_gpu.gpudata
  <pycuda._driver.DeviceAllocation object at 0x8c0d454>

Let's for example call the function from the section before::

  >>> y_gpu = gpuarray.zeros(5,float32)
  >>> info = array([5]).astype(int32)
  >>> info_gpu = gpuarray.to_gpu(info)
  >>> func(y_gpu.gpudata,x_gpu.gpudata,info_gpu.gpudata, block = (32,1,1), grid = (4,1))
  >>> y_gpu
  array([-1.09917879,  0.56061697, -0.19573164, -4.29430866, -2.519032  ], dtype=float32)
  >>> 2*x_gpu
  array([-1.09917879,  0.56061697, -0.19573164, -4.29430866, -2.519032
  >>> ], dtype=float32)

``gpuarrays`` can be bound to textures too::

  >>> x_gpu.bind_to_texref_ext(MyTexture)

.. _mpi_and_pycuda_ref:

Using MPI4Py and PyCUDA together
--------------------------------

I give here a short example how to use this, to get PyCUDA
working with MPI4Py. We initialize as many threads, as graphic
cards available (in this case 4) and do something on that devices. 
Every thread is working on one device.

::

  from mpi4py import MPI
  import pycuda.driver as cuda
  
  cuda.init() #init pycuda driver
  
  from pycuda import gpuarray
  from numpy import float32, array
  from numpy.random import randn as rand
  import time
  
  comm = MPI.COMM_WORLD
  rank = comm.rank
  root = 0
  
  nr_gpus = 4
  
  sendbuf = []
  
  N = 2**20*nr_gpus
  K = 1000

  if rank == 0:
      x = rand(N).astype(float32)*10**16
      print "x:", x
      
      t1 = time.time()
      
      sendbuf = x.reshape(nr_gpus,N/nr_gpus)
  
  if rank > nr_gpus-1:
      raise ValueError("To few gpus!")

 
  current_dev = cuda.Device(rank) #device we are working on
  ctx = current_dev.make_context() #make a working context
  ctx.push() #let context make the lead

  #recieve data and port it to gpu:
  x_gpu_part = gpuarray.to_gpu(comm.scatter(sendbuf,root))

  #do something...
  for k in xrange(K):
    x_gpu_part = 0.9*x_gpu_part

  #get data back:
  x_part = (x_gpu_part).get()

  ctx.pop() #deactivate again
  ctx.detach() #delete it
  
  recvbuf=comm.gather(x_part,root) #recieve data
  
  if rank == 0:
      x_doubled = array(recvbuf).reshape(N) 
      t2 = time.time()-t1
  
      print "doubled x:", x_doubled
      print "time nedded:", t2*1000, " ms " 


.. rubric:: Links

.. [#] http://www.nvidia.com/object/cuda_home_new.html 
.. [#] http://mathema.tician.de/software/pycuda
.. [#] http://trac.sagemath.org/sage_trac/ticket/10010
.. [#] http://documen.tician.de/pycuda/
.. [#] http://wiki.tiker.net/PyCuda

