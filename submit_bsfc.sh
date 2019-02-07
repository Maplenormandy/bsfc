#!/bin/bash 

#SBATCH -J bsfc_run_1101014019
#SBATCH -N 3
#SBATCH -n 96
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive
#SBATCH --time=0:29:59
#SBATCH --partition sched_mit_psfc_short
#SBATCH --output %j.out

#Load modules
module purge 
module use /home/sciortino/modulefiles
module load sciortino/bsfc

# Set inter-node communication standard:
export I_MPI_FABRICS=shm:tcp

# print out date and time 
date

# Print a few job details: 
echo "SLURM_JOB_ID: " $SLURM_JOB_ID
echo "SLURM_JOB_CPUS_PER_NODE: " $SLURM_JOB_CPUS_PER_NODE
echo "SLURM_CLUSTER_NAME: " $SLURM_CLUSTER_NAME
echo "SLURM_MEM_PER_CPU: " $SLURM_MEM_PER_CPU
echo "SLURM_TASKS_PER_NODE: " $SLURM_TASKS_PER_NODE
echo "SLURM_JOB_NAME: " $SLURM_JOB_NAME
echo "SLURM_NTASKS: " $SLURM_NTASKS

# ===========================
shot=1101014019  #1120914036 # 1121002022 # 1101014029
nsteps=100000

mpirun python bsfc_run_mpi.py $shot $nsteps 

