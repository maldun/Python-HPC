# -*- coding: utf-8 -*-
from mpi4py import MPI
comm = MPI.COMM_WORLD
print("hello world")
print("my rank is: %d"%comm.rank)

#start with $SAGE_ROOT/local/bin/mpirun sage -python mpi.py
