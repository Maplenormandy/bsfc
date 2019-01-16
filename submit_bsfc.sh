#!/bin/bash 

#SBATCH -J bsfc_run_1101014019
#SBATCH -N 10
#SBATCH -n 320
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive
#SBATCH --time=11:29:59
#SBATCH --partition sched_mit_psfc #_short
#SBATCH --output %j.out
#SBATCH --mail-type=END 
#SBATCH --mail-user=sciortino@psfc.mit.edu 
#######SBATCH --ntasks-per-node=8
####SBATCH --qos=psfc_24h

# Don't load modules since they were loaded in the .bashrc already
 
# Set inter-node communication standard:
export I_MPI_FABRICS=shm:tcp

# print out date and time 
date

# Copy the virtual environment to the /tmp directory of the assigned node
#srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c ./bsfc_env_setup.sh

# Activate virtual environment:
#. /tmp/bsfc_env/bin/activate
#echo "********** Activated Virtual Environment **********"
module load sciortino/bsfc

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

