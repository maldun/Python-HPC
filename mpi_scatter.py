# -*- coding: utf-8 -*-
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
