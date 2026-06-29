#!/bin/bash -l
# job name
#SBATCH -J bench_vtune
# standard output
#SBATCH -o %x.out
#SBATCH --partition=short
# number of nodes
#SBATCH --nodes=1
# number of openmp threads
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00

BUILD_DIR=/absolute/path/to/genex/build/directory
module purge
source ${BUILD_DIR}/toolchain.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun vtune -collect memory-access \
           -r vtune_results \
           -- ${BUILD_DIR}/bin/benchmark-operators -i params_in.txt
