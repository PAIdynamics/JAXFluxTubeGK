#!/bin/bash -l
#SBATCH -J genex_utests_cpu
#SBATCH -o utests_cpu.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --mail-type=ALL          # Uncomment, enter your email and remove
##SBATCH --mail-user=<YOUR_EMAIL> # these comments to receive job notifications

module purge
source ../toolchain.sh
# set stacksize to unlimited, otherwise some tests might fail
ulimit -s unlimited

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

if [[ $MACH_NAME == "IZUM Vega" ]] && \
   [[ $GENEX_C_COMPILER == "icx" ]]; then
    module load PMIx/4.1.2-GCCcore-11.3.0
    module unload OpenSSL/1.1
    export I_MPI_PMI_LIBRARY=/cvmfs/sling.si/modules/el7/software/PMIx/4.1.2-GCCcore-11.3.0/lib/libpmix.so
    export SLURM_MPI_TYPE=pmix_v3
fi

if [[ "$MACH_NAME" == "MPCDF Viper-CPU" ]]; then
    module is-loaded openmpi
    have_openmpi=$?
    if [[ "$have_openmpi" == 0 ]]; then
	UCX_PATH=/mpcdf/soft/RHEL_9/packages.norpm/ucx-1.16.0_gsys_no_march
	export LD_LIBRARY_PATH=${UCX_PATH}/lib/ucx:${UCX_PATH}/lib:${LD_LIBRARY_PATH}
	echo ""
	echo "====================================================================="
	echo "== Changing LD_LIBRARY_PATH explicitly to ucx-1.16.0_gsys_no_march =="
	echo "== Remove this in unit-testing/submit_utests_cpu.sh if possible.   =="
	echo "====================================================================="
	echo ""
    fi
fi

ctest --test-dir ../src/ --output-on-failure -E "utests-gpu"
