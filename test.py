import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print "My rank: ", rank

import time as time_

time_.sleep(60)
