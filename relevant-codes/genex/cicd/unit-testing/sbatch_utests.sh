#!/usr/bin/env bash

# Default values
print_command_only=false
print_help=false
run_gpu=false
slurm_wait=false
kill_terminal=false

# Check arguments
for arg in "${@:Q}"; do
    if [[ $arg == "-h" ]]; then
        print_help=true

    elif [[ $arg == "-p" ]]; then
        print_command_only=true

    elif [[ $arg == "--gpu" ]]; then
        run_gpu=true

    elif [[ $arg == "--wait" ]]; then
        slurm_wait=true

    elif [[ $arg == "--kill" ]]; then
        kill_terminal=true

    fi
done

# Print help
if ${print_help}; then
    echo ""
    echo "sbatch_utest.sh runs unit testing of GENE-X with SLURM sbatch."
    echo "This needs to be executed from <BUILD_DIR>/unit-testing/."
    echo ""
    echo "> source sbatch_utest.sh <options>"
    echo ""
    echo "  Optional arguments <options>:"
    echo "    - -h    : Open help messages"
    echo "    - -p    : Only print the command without executing it"
    echo "    - --gpu  : Activate build version with GPU features"
    echo "    - --wait : Instruct SLURM to wait until the job is finished"
    echo "    - --kill : Kill terminal if the job fails"
    echo ""
    return 1;
fi

# Get the SLURM billing account
if [[ -z "${GXBILL}" ]]; then
    BILLING_ACCOUNT=""
else
    BILLING_ACCOUNT="-A ${GXBILL}"
fi

# Check if user wants to wait until SLURM job is finished
utests_command="sbatch --export=MACH_NAME=\"$MACH_NAME\",PROF_DIR=\"$PROF_DIR\""
if ${slurm_wait}; then
    utests_command="${utests_command} --wait"
fi

# Command for unit testing on CPU
utests_cpu="$CONFIX_UTESTS_SLURM_CPU $BILLING_ACCOUNT submit_utests_cpu.sh"

# Command for unit testing on GPU
utests_gpu="$CONFIX_UTESTS_SLURM_GPU $BILLING_ACCOUNT submit_utests_gpu.sh"

# Choose between CPU or GPU
if ${run_gpu}; then
    utests_command="${utests_command} ${utests_gpu}"
    suffix="gpu"
else
    utests_command="${utests_command} ${utests_cpu}"
    suffix="cpu"
fi

# Print out the command for visibility
echo -e "${BOLDBLUE}${utests_command}${NC}"

# Remove existing output file
if [ -f utests_${suffix}.out ]; then
    rm utests_${suffix}.out
fi

# Execute the command
if ! ${print_command_only}; then
    if ${slurm_wait}; then
        if eval ${utests_command}; then
            cat utests_${suffix}.out

        else
            cat utests_${suffix}.out
            if ${kill_terminal}; then
                exit 1
            fi

        fi

    else
        eval ${utests_command}
    fi
fi
