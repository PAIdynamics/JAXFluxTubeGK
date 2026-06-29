submodule (test_params_m) test_params_gpu_offload_s
    ! Submodule that contains helpers for unit tests to initialize
    ! GPU offloading features with non-default parameters.
    use logger_m, only: logger_get_debug_channel
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_gpu_offload(comm, rank, print_messages, &
                                             debug, swap_mesh_members)
        !! Initialize the simulation with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        logical, optional, intent(in) :: print_messages
        logical, optional, intent(in) :: debug
        logical, optional, intent(in) :: swap_mesh_members

        character(len=16) :: genex_utest_gpu_backend, &
                             genex_utest_parallax_gpu_backend
        character(len=:), allocatable :: outmsg, px_outmsg, params_file
        logical :: use_gpu_local, use_px_gpu_local, use_px_data_local, &
                   swap_local
        integer :: gpu_backend_local, px_gpu_backend_local
        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        use_gpu_local         = get_use_gpu_offload()
        gpu_backend_local     = get_gpu_offload_backend()
        use_px_gpu_local      = get_use_parallax_gpu_offload()
        px_gpu_backend_local  = get_parallax_gpu_offload_backend()
        use_px_data_local     = get_use_parallax_gpu_data_explicit()
        swap_local            = get_swap_mesh_members()

        if(present(swap_mesh_members)) swap_local = swap_mesh_members

        params_file = trim(get_test_params_file())//".txt"

#ifdef ENABLE_GPU

        ! Fetch the environment variable dedicated to GPU unit testing
        call get_environment_variable("GENEX_UTEST_GPU_BACKEND", &
                                      genex_utest_gpu_backend)

        select case(trim(genex_utest_gpu_backend))
            case("CXX")
                ! Set the unit test to run the GPU types but on CPU
                outmsg = "Unit tests will run with the GPU types &
                         &with GPU_OFFLOAD_CPU."

                use_gpu_local     = .true.
                gpu_backend_local = GPU_OFFLOAD_CPU

            case("OPENACC")
                ! Set the unit test to run the GPU types on GPU via OpenACC
                outmsg = "Unit tests will run with the GPU types &
                         &with GPU_OFFLOAD_ACC."

                use_gpu_local     = .true.
                gpu_backend_local = GPU_OFFLOAD_ACC

            case("OPENMPX")
                ! Set the unit test to run the GPU types on GPU
                ! via OpenMP offload
                outmsg = "Unit tests will run with the GPU types &
                         &with GPU_OFFLOAD_OMPX."

                use_gpu_local     = .true.
                gpu_backend_local = GPU_OFFLOAD_OMPX

            case("CUDA")
                ! Set the unit test to run the GPU types on GPU via CUDA
                outmsg = "Unit tests will run with the GPU types &
                         &with GPU_OFFLOAD_CUDA."

                use_gpu_local     = .true.
                gpu_backend_local = GPU_OFFLOAD_CUDA

            case default
                ! Set the unit test to run the CPU types (default)
                outmsg = "Unit tests will run with the CPU types."

        end select

#else

        ! Set the unit test to run the CPU types (default)
        outmsg = "Unit tests will run with the CPU types."

#endif

#ifdef ENABLE_PARALLAX_GPU

        ! Fetch the environment variable dedicated to PARALLAX GPU features in
        ! GENE-X unit testing
        call get_environment_variable("GENEX_UTEST_PARALLAX_GPU_BACKEND", &
                                      genex_utest_parallax_gpu_backend)

        select case(trim(genex_utest_parallax_gpu_backend))
            case("CXX")
                ! Set PARALLAX with C++ solver on CPU
                px_outmsg = "PARALLAX C++ solver is used on CPU."

                use_px_gpu_local      = .true.
                px_gpu_backend_local  = PARALLAX_BACKEND_CPU

            case("CUDA")
                ! Set PARALLAX with C++ solver on GPU
                px_outmsg = "PARALLAX C++ solver is used on GPU."

                use_px_gpu_local      = .true.
                px_gpu_backend_local  = PARALLAX_BACKEND_GPU

            case("CUDA_ADVANCED")
                ! Set the explicit GPU data management of PARALLAX GPU solver
                px_outmsg = "PARALLAX CUDA solver with explicit GPU data &
                            &management is used."

                use_px_gpu_local      = .true.
                px_gpu_backend_local  = PARALLAX_BACKEND_GPU
                use_px_data_local     = .true.

            case default
                ! Set PARALLAX with Fortran solver on CPU
                px_outmsg = "PARALLAX Fortran solver is used."

        end select

#else

        ! Set PARALLAX with Fortran solver on CPU
        px_outmsg = "PARALLAX Fortran solver is used."

#endif

        ! Print out messages about GPU offloading parameters if requested
        if(present(print_messages)) then
            ! Only master process prints out the message
            if(print_messages .and. rank == 0) then
                write(logger_get_debug_channel(), "(A,I1)") &
                    trim(outmsg) // " " // trim(px_outmsg)
            endif
        endif

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=params_file, &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_gpu_offload")

            call write_nml(iunit, "params_gpu_offload", &
                           "use_gpu_offload", use_gpu_local)
            call write_nml(iunit, "params_gpu_offload", &
                           "gpu_offload_backend", gpu_backend_local)
            call write_nml(iunit, "params_gpu_offload", &
                           "use_parallax_gpu_offload", use_px_gpu_local)
            call write_nml(iunit, "params_gpu_offload", &
                           "parallax_gpu_offload_backend", px_gpu_backend_local)
            call write_nml(iunit, "params_gpu_offload", &
                           "use_parallax_gpu_data_explicit", use_px_data_local)
            call write_nml(iunit, "params_gpu_offload", &
                           "swap_mesh_members", swap_local)

            call close_nml(iunit, "params_gpu_offload")

            close(iunit)

        endif

        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_gpu_offload(params_file)
        call mpi_barrier(comm, ierr)

        ! Check the sanity of GPU functionalities
        if(present(debug)) then
            if(debug) then
                call check_gpu_functionalities()
            endif
        endif

    end subroutine

end submodule
