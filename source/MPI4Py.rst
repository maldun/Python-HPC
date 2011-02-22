MPI4Py
======

MPI4Py is a Python module for calling the MPI API.
For more information and detailed documentation I refer to
the official MPI4Py documentation [#]_
Let's start with the MPI Hello World program in Python::

  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  print("hello world")
  print("my rank is: %d"%comm.rank)

As it can be seen the API is quite similar to the normal MPI API in C.
First we save this file as *mpi.py*.
To call now our parallized version of the Hello World program simply 
call the Python Interpreter with MPI::

  $ ./where/mpi/is/installed/mpirun -n <nr_processes> python mpi.py

(If you use Sage, you have to install the openmpi package, and then you
can find mpirun in ``SAGE_LOCAL/bin/``)
I for example use Sage, and this would look like this::

  $ $SAGE_ROOT/local/bin/mpirun -n 4 sage -python mpi.py
  hello world
  my rank is: 2
  hello world
  my rank is: 0
  hello world
  my rank is: 1
  hello world
  my rank is: 3

Here another example: We generate an array with a thread which is
currently our main thread. Then we distribute it over all threads we
called::

  from mpi4py import MPI
  import numpy
  import time

  comm = MPI.COMM_WORLD
  rank = comm.rank

  sendbuf=[]
  root=0
  if rank==0:
      m=numpy.random.randn(comm.size,comm.size)
      print(m)
      sendbuf=m
      t1 = time.time()

  v=MPI.COMM_WORLD.scatter(sendbuf,root)

  print(rank,"I got this array:")
  print(rank,v)

  v=v*2

  recvbuf=comm.gather(v,root)

  if rank==0:
    t2 = time.time()
    print numpy.array(recvbuf)
    print "time:", (t2-t1)*1000, " ms "

This snippet produces this output::

  $ $SAGE_ROOT/local/bin/mpirun -n 3 sage -python mpi_scatter.py
  [[ -5.90596754e-04   4.21504158e-02   2.11213337e-01]
   [  9.67314022e-01  -2.16766512e+00   1.00552694e+00]
   [  1.37283086e+00  -2.29582623e-01   2.88653028e-01]]
  (0, 'I got this array:')
  (0, array([-0.0005906 ,  0.04215042,  0.21121334]))
  (1, 'I got this array:')
  (1, array([ 0.96731402, -2.16766512,  1.00552694]))
  (2, 'I got this array:')
  (2, array([ 1.37283086, -0.22958262,  0.28865303]))
  [[ -1.18119351e-03   8.43008316e-02   4.22426674e-01]
   [  1.93462804e+00  -4.33533025e+00   2.01105389e+00]
   [  2.74566171e+00  -4.59165246e-01   5.77306055e-01]]
  time: 3.59892845154  ms 

For further examples I refer to the Sage tutorial for scientific
computing.  [#]_
**Note** The last time I checked the tutorial, it was outdated.
If you need a corrected version, I posted one on Sage trac [#]_.


.. rubric:: Links

.. [#] http://mpi4py.scipy.org/docs/usrman/index.html
.. [#] http://www.sagemath.org/doc/numerical_sage/mpi4py.html
.. [#] http://trac.sagemath.org/sage_trac/attachment/ticket/10566/mpi4py.rst
