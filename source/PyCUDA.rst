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

Get your CUDA code working in Python
------------------------------------

Similar to ::ref::`weave_ref` we can write CUDA code as string in
Python and then


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
.. [#] wikilink

