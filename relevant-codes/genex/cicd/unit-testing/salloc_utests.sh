#!/usr/bin/env bash

# Default values
print_command_only=false
print_help=false
run_gpu=false
mpcdf_gpu=false

# Check arguments
for arg in "${@:Q}"; do
    if [[ $arg == "-h" ]]; then
        print_help=true

    elif [[ $arg == "-p" ]]; then
        print_command_only=true

    elif [[ $arg == "--gpu" ]]; then
        run_gpu=true

    fi
done

# Print help
if ${print_help}; then
    echo ""
    echo "salloc_utest.sh runs unit testing of GENE-X with SLURM salloc."
    echo "This needs to be executed from <BUILD_DIR>/unit-testing/."
    echo ""
    echo "> source salloc_utest.sh <options>"
    echo ""
    echo "  Optional arguments <options>:"
    echo "    - -h   : Open help messages"
    echo "    - -p   : Only print the command without executing it"
    echo "    - --gpu : Activate build version with GPU features"
    echo ""
    return 1;
fi

# Check if it's a GPU job in MPCDF Raven and MPCDF Viper-GPU
if ${run_gpu}; then
    if [[ $MACH_NAME == "MPCDF Raven" ]]; then
        mpcdf_gpu=true
        mpcdf_gpu_slurm="--constraint=gpu --partition=gpu --gres=gpu:a100:1"

        utests_part_1="-I 1,14"
        utests_part_2="-I 15,17"
        utests_part_3="-I 18,"
    elif [[ $MACH_NAME == "MPCDF Viper-GPU" ]]; then
        mpcdf_gpu=true
        mpcdf_gpu_slurm="--constraint=apu --partition=apu --gres=gpu:1"

        utests_part_1="-I 1,10"
        utests_part_2="-I 11,"
        utests_part_3=""
    fi
fi

# Temporarily unload anaconda module in Leonardo
if [[ $MACH_NAME == "CINECA Leonardo" ]]; then
    python_mod=$(module -t list | grep anaconda)
    if [ -n $python_mod ]; then
        module unload $python_mod
    fi
fi

# Get the SLURM billing account
if [[ -z "${GXBILL}" ]]; then
    BILLING_ACCOUNT=""
else
    BILLING_ACCOUNT="-A ${GXBILL}"
fi

if ! ${mpcdf_gpu}; then

    # Command for unit testing on CPU
    utests_cpu="salloc -N 1 -n 8 -t 01:30:00 --mem=32G -J genex_utests"
    utests_cpu="$utests_cpu $CONFIX_UTESTS_SLURM_CPU $BILLING_ACCOUNT ctest"
    utests_cpu="$utests_cpu --test-dir ../src/ --output-on-failure"
    utests_cpu="$utests_cpu -E \"utests-gpu\""

    # Command for unit testing on GPU
    utests_gpu="salloc -N 1 -n 8 -t 01:00:00 --mem=32G -J genex_gputests"
    utests_gpu="$utests_gpu $CONFIX_UTESTS_SLURM_GPU $BILLING_ACCOUNT ctest"
    utests_gpu="$utests_gpu --test-dir ../src/ --output-on-failure"
    utests_gpu="$utests_gpu -R \"utests-gpu\""

    # Choose between CPU or GPU
    if ${run_gpu}; then
        suffix="gpu"
        utests_command="${utests_gpu} -O utests_${suffix}.out"
    else
        suffix="cpu"
        utests_command="${utests_cpu} -O utests_${suffix}.out"
    fi

    # Print out the command for visibility
    echo -e "${BOLDBLUE}${utests_command}${NC}"

    # Execute the command
    if ! ${print_command_only}; then
        eval ${utests_command}
    fi

else
    # Command for unit testing on GPU
    utests_gpu="salloc -N 1 -n 8 -t 00:30:00 --mem=32G -J genex_gputests"
    utests_gpu="$utests_gpu $mpcdf_gpu_slurm"
    utests_gpu="$utests_gpu ctest --test-dir ../src/ --output-on-failure"
    utests_gpu="$utests_gpu -R \"utests-gpu\""

    # Print out the command for visibility
    if [ -n "${utests_part_1}" ]; then
        utests_command="${utests_gpu} ${utests_part_1} -O utests_gpu_1.out"
        echo -e "${BOLDBLUE}${utests_command}${NC}"

        # Execute the command
        if ! ${print_command_only}; then
            eval ${utests_command}
        fi
    fi

    # Print out the command for visibility
    if [ -n "${utests_part_2}" ]; then
        utests_command="${utests_gpu} ${utests_part_2} -O utests_gpu_2.out"
        echo -e "${BOLDBLUE}${utests_command}${NC}"

        # Execute the command
        if ! ${print_command_only}; then
            eval ${utests_command}
        fi
    fi

    # Print out the command for visibility
    if [ -n "${utests_part_3}" ]; then
        utests_command="${utests_gpu} ${utests_part_3} -O utests_gpu_3.out"
        echo -e "${BOLDBLUE}${utests_command}${NC}"

        # Execute the command
        if ! ${print_command_only}; then
            eval ${utests_command}
        fi
    fi

fi

# Load back the unloaded anaconda module in leonardo
if [[ $MACH_NAME == "CINECA Leonardo" ]] && [[ -n $python_mod ]]; then
    module load $python_mod
fi
