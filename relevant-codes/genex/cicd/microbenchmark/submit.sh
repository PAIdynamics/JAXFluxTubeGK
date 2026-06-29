#!/bin/bash -l
# job name
#SBATCH -J bench
# standard output
#SBATCH -o %x.out
#SBATCH --partition=express
# number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# number of openmp threads
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00

BUILD_DIR=/absolute/path/to/genex/build/directory
module purge
source ${BUILD_DIR}/toolchain.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=verbose

# Run the program:
srun ${BUILD_DIR}/bin/benchmark-operators -i params_in.txt
