#!/bin/bash -l
# job name
#SBATCH -J bench_advisor
# standard output
#SBATCH -o %x.out
#SBATCH --partition=short
# number of nodes
#SBATCH --nodes=1
# number of openmp threads
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00

REPO_DIR=/absolute/path/to/genex/repository/directory
BUILD_DIR=/absolute/path/to/genex/build/directory
module purge
source ${BUILD_DIR}/toolchain.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=verbose

srun advixe-cl \
    --collect=survey \
    --search-dir src:r=${REPO_DIR}/src \
    --project-dir Roofline \
    -- ${BUILD_DIR}/bin/benchmark-operators -i params_in.txt
sleep 1

srun advixe-cl \
    --collect=tripcounts \
    -flop --no-trip-counts \
    --search-dir src:r=${REPO_DIR}/src \
    --project-dir Roofline \
    -- ${BUILD_DIR}/bin/benchmark-operators -i params_in.txt
sleep 1
