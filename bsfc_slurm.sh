#!/bin/bash 

#SBATCH -J bsfc_run_1101014029
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH --partition sched_mit_psfc
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

# Activate virtual environment:
. ~/cenv/bin/activate
echo "********** Activated Virtual Environment **********"

# Print a few job details: 
echo "SLURM_JOB_ID: " $SLURM_JOB_ID
echo "SLURM_JOB_CPUS_PER_NODE: " $SLURM_JOB_CPUS_PER_NODE
echo "SLURM_CLUSTER_NAME: " $SLURM_CLUSTER_NAME
echo "SLURM_MEM_PER_CPU: " $SLURM_MEM_PER_CPU
echo "SLURM_TASKS_PER_NODE: " $SLURM_TASKS_PER_NODE
echo "SLURM_JOB_NAME: " $SLURM_JOB_NAME
echo "SLURM_NTASKS: " $SLURM_NTASKS

# ===========================
shot=1101014029  #1120914036
option=3
nsteps=50000

python bsfc_run.py $shot $SLURM_NTASKS $option $nsteps

# run python command twice to plot after running this script