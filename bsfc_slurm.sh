#!/bin/bash 

#SBATCH -J bsfc_test
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --mem-per-cpu=4000
#SBATCH --exclusive
#SBATCH --time=11:59:00
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

python bsfc_main.py $SLURM_NTASKS

# Run with MPI:
# mpirun python /tmp/bayesimp_dir/mitim_settings.py $SLURM_JOB_NAME $shot $n_live_points $sampling_efficiency $SLURM_ARRAY_TASK_ID

# Rename output file with job name
# mv $SLURM_JOB_ID.out $SLURM_JOB_NAME.out
#rm -r cenv_$SLURM_JOB_ID
#rm $SLURM_JOB_ID.out

# Output the loaded environment
# echo ""
# echo ""
# echo ""
# env