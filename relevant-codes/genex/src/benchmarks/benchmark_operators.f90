program benchmark_operators
    !! Benchmark program for GENE-X operators
#ifdef ADV_ANNOTATE
    use advisor_annotate
#endif
    use mpi
    use, intrinsic :: iso_fortran_env
    use genex_build_info_m, only: git_hash, git_date
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_BENCHMARKS, &
                                    GENEX_WRN_FILE_NOT_EXIST
    use logger_m, only: logger_set_debug_channel, logger_get_info_channel, &
                        logger_get_debug_channel
    use dcomm_handler_m, only: dcomm_handler_t
    use genex_fortran_env_m, only: GP
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use mesh_5d_m, only: mesh_5d_t
    use data_array_m, only: data_array_2d_t, data_array_4d_t, data_array_5d_t
    use params_mesh_m, only: get_equilibrium_type, get_spacing_RZ, &
                             get_n_points_phi, get_n_points_vp, &
                             get_n_points_mu, get_quad_type_vp, &
                             get_grid_type_mu, get_use_bsg, get_use_vspectral
    use params_bsg_m, only: get_num_bsg_blocks, get_bsg_interp_order
    use params_gpu_offload_m, only: get_use_gpu_offload, &
                                    get_gpu_offload_backend, &
                                    check_gpu_functionalities, &
                                    GPU_OFFLOAD_CPU, &
                                    GPU_OFFLOAD_ACC, &
                                    GPU_OFFLOAD_OMPX
    use params_parallelization_m, only: get_n_procs_phi, get_n_procs_vp, &
                                        get_n_procs_mu, get_n_procs_sp
    use params_m, only: set_parameter_file, read_parameter_file
    use file_handling_m, only: file_exists

    ! From PARALLAX
    use equilibrium_factory_m, only: SALPHA
    use screen_io_m, only: set_fci_core_stdout => set_stdout

    implicit none

    type(op_set_uniform_cpu_t), allocatable :: op_set_uniform

    type(mesh_5d_t), target, allocatable :: mesh
    type(dcomm_handler_t), allocatable, target :: dcomm_handler
    integer :: n_rz, n_phi, n_vp, n_mu, n_sp
    integer :: n_t, t
    integer :: ierr, thread_level

    integer :: i, argc, debug_channel
    character(len=256), allocatable, dimension(:) :: argv
    character(len=256) :: parameter_file
    logical :: parameter_check
    character(len=64) :: runtime_version_string
    character(len=256) :: out_dir
    logical :: read_mesh_from_file
    character(:), allocatable :: mesh_filename

    ! By default performance collection is started, we disable it here and
    ! only start it before the computational regions.
    call stop_perf_collection()

    ! Process command line arguments
    argc = command_argument_count()
    if(argc > 7) then
        call print_usage()
        stop
    endif

    ! Retrieve all arguments
    allocate(argv(argc))
    do i = 1, argc
        call get_command_argument(i, argv(i))
    enddo

    ! Parse the optional arguments
    ! NOTE : -cxx, -acc, -ompx options are temporary implementations. This will
    !        be replaced by a parameter file read routine as part of
    !        params_gpu_offload_m.
    parameter_file = "params_in.txt"
    parameter_check = .false.
    out_dir = ""
    n_t = 20
    i = 1
    do while(i <= argc)
        select case(argv(i))
        case("-h")
            call print_usage()
            stop
        case("-c")
            parameter_check = .true.
        case("-o")
            if(i > argc - 1) then
                ! The output directory has to be specified after -o
                call print_usage()
                stop
            endif
            out_dir = trim(argv(i + 1))
            if(out_dir(1:1) == "-") then
                ! The output directory may not be an optional variable
                call print_usage()
                stop
            endif
            i = i + 1
        case("-i")
            if(i > argc - 1) then
                ! The parameter file has to be specified after -i
                call print_usage()
                stop
            endif
            parameter_file = trim(argv(i + 1))
            i = i + 1
        case("-n")
            if(i > argc - 1) then
                ! The number of timesteps has to be specified after -n
                call print_usage()
                stop
            endif
            read (argv(i + 1), "(I10)") n_t
            i = i + 1
        end select
        i = i + 1
    end do

    deallocate(argv)

    ! In order to prevent compiler optimization we need to print data in every
    ! benchmark iteration. We print the data to the debug_channel of the logger
    ! and redirect the output to a log file called debug_log.txt
    open(newunit=debug_channel, file="debug_log.txt", status='replace', &
         action='write', iostat=ierr)
    if(ierr /= 0) then
        call handle_error("Could not create debug log file!", &
                          GENEX_ERR_BENCHMARKS, __LINE__, __FILE__)
    endif
    call logger_set_debug_channel(debug_channel)
    ! The redirection of the standard out of parallax into the debug unit is
    ! currently disabled because it causes segmentation faults upon write on
    ! cobra. The root of the problem is currently not known.
    !call set_fci_core_stdout(debug_channel)

    if(parameter_check) then
        call check_parameters(trim(parameter_file))
        close(debug_channel)
        stop
    endif

    if(trim(parameter_file) /= "") then
        ! Read the namelists from the parameter file
        call set_parameter_file(parameter_file)
        call read_parameter_file()
    endif

    ! Get message string for each GPU offloading backend
    runtime_version_string = "pure Fortran on CPU"
    if(get_use_gpu_offload()) then
        select case (get_gpu_offload_backend())
        case(GPU_OFFLOAD_CPU)
            runtime_version_string = "Fortran/C++ hybrid on CPU &
                                     &via OpenMP if available"
        case(GPU_OFFLOAD_ACC)
            runtime_version_string = "Fortran/C++ hybrid on GPU &
                                     &via OpenACC if available"
        case(GPU_OFFLOAD_OMPX)
            runtime_version_string = "Fortran/C++ hybrid on GPU &
                                     &via OpenMP offload if available"
        end select
    endif

    ! Initialize mpi
#ifdef _OPENMP
    call MPI_Init_thread(MPI_THREAD_FUNNELED, thread_level, ierr)
    if(thread_level /= MPI_THREAD_FUNNELED) then
        call handle_error("Hybrid MPI + OpenMP not supported!", &
                          GENEX_ERR_BENCHMARKS, __LINE__, __FILE__, &
                          additional_info=error_info_t(&
                            "Thread level provided was:", [thread_level]))
    endif
#else
    call MPI_Init(ierr)
#endif

    ! Check if GPUs are found and can be used as intended
    call check_gpu_functionalities()

    ! Allocate and create the simulation mesh
    allocate(dcomm_handler)

    if(parameter_file /= "") then
        call dcomm_handler%initialize(MPI_COMM_WORLD, get_n_procs_phi(), &
                                      get_n_procs_vp(), get_n_procs_mu(), &
                                      get_n_procs_sp())
    else
        call dcomm_handler%initialize(MPI_COMM_WORLD, 1, 1, 1, 1)
    endif

    ! Welcome the user
    write(logger_get_info_channel(), *) "Welcome to the benchmark program for &
                                        &GENE-X operators!"
    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "   You are running commit ", git_hash
    write(logger_get_info_channel(), *) "   From ", git_date
    write(logger_get_info_channel(), *) ""

    ! Prints out the number of processes
    write(logger_get_info_channel(), *) "Number of procs used"
    write(logger_get_info_channel(), *) "    in PHI direction: ", &
                                        dcomm_handler%get_n_procs_phi()
    write(logger_get_info_channel(), *) "    in VP  direction: ", &
                                        dcomm_handler%get_n_procs_vp()
    write(logger_get_info_channel(), *) "    in MU  direction: ", &
                                        dcomm_handler%get_n_procs_mu()
    write(logger_get_info_channel(), *) "    in SP  direction: ", &
                                        dcomm_handler%get_n_procs_sp()

    mesh_filename = trim(out_dir) // "mesh.nc"

    if(file_exists(mesh_filename)) then
        read_mesh_from_file = .true.
    else
        call handle_error("Attempt to read mesh from file which does &
                          &not exist! Initializing mesh from scratch.",&
                          GENEX_WRN_FILE_NOT_EXIST, __LINE__, __FILE__,&
                          additional_info=error_info_t(&
                            "Filename was: "//mesh_filename))
        read_mesh_from_file = .false.
    endif

    ! We simulate the typical load of a single node, i.e. it contains all
    ! points in the plane and due to parallelization only 1 phi plane and
    ! some points in velocity space.
    ! To benchmark the LBD operator we have to use a quadratic mu grid.
    allocate(mesh)
    call mesh%initialize(dcomm_handler, read_mesh_from_file, mesh_filename)

    ! Allocate and initialize global data
    n_rz = mesh%size_RZ()
    n_phi = mesh%size_phi()
    n_vp = mesh%size_vp()
    n_mu = mesh%size_mu()
    n_sp = mesh%size_sp()

    ! Prints out mesh properties
    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Mesh properties"
    write(logger_get_info_channel(), *) "    Equilibrium type     : ", &
                                        mesh%equi_type()
    write(logger_get_info_channel(), *) "    Type of vp quadrature: ", &
                                        mesh%quad_type_vp()
    write(logger_get_info_channel(), *) "    Type of mu grid      : ", &
                                        mesh%grid_type_mu()
    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Number of points used:", &
                        int(n_rz, kind=INT64) * int(n_phi, kind=INT64) * &
                        int(n_vp, kind=INT64) * int(n_mu, kind=INT64) * &
                        int(n_sp, kind=INT64)
    write(logger_get_info_channel(), *) "    in RZ  direction: ", n_rz
    write(logger_get_info_channel(), *) "    in PHI direction: ", n_phi
    write(logger_get_info_channel(), *) "    in VP  direction: ", n_vp
    write(logger_get_info_channel(), *) "    in MU  direction: ", n_mu
    write(logger_get_info_channel(), *) "    in SP  direction: ", n_sp

    ! Prints out number of timesteps
    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Number of timesteps: ", n_t

    allocate(op_set_uniform)
    call op_set_uniform%initialize()

    ! Run the different operator benchmarks
    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_set_uniform_t"
    call benchmark_set_uniform()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_add_t"
    call benchmark_axpy()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_lin_comb_t"
    call benchmark_lin_comb()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_copy_t"
    call benchmark_copy()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_mom_maxwells_eq_t"
    call benchmark_mom_maxwells_eq()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_mom_0d_t"
    call benchmark_mom_0d()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_mom_2d_t"
    call benchmark_mom_2d()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) "Testing op_rhs_vlasov_eq_static_t &
                                        &with ", runtime_version_string
    call benchmark_rhs_vlasov_eq_static()

    write(logger_get_info_channel(), *) ""
    write(logger_get_info_channel(), *) &
          "Testing op_rhs_vlasov_eq_dynamic_t &
          &with ", runtime_version_string
    call benchmark_rhs_vlasov_eq_dynamic()

    if(.not. get_use_vspectral()) then
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "Testing op_mom_coll_cpu_t"
        call benchmark_mom_coll("op_mom_coll_cpu_t")
    else
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "Testing op_mom_coll_vspec_cpu_t"
        call benchmark_mom_coll("op_mom_coll_vspec_cpu_t")
    end if

    if(.not. get_use_vspectral()) then
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "Testing op_coll_bgk_t"
        call benchmark_coll("bgk")
    end if

    if(get_quad_type_vp() == "midpoint") then
    if(get_grid_type_mu() == "quadratic") then
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "Testing op_coll_lbd_t"
        call benchmark_coll("lbd")

        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "Testing op_coll_lorentz_t"
        call benchmark_coll("lorentz")
    end if
    end if

    deallocate(mesh)
    deallocate(op_set_uniform)
    deallocate(dcomm_handler)

    ! Finalize mpi
    call MPI_Finalize(ierr)
    close(debug_channel)

contains

    subroutine start_perf_collection()
        !! Starts the collection of performance data.
        !! Uses Intel Advisor annotations for that. If library was not found,
        !! this subroutine will do nothing.
#ifdef ADV_ANNOTATE
        call annotate_disable_collection_pop()
#endif
    end subroutine

    subroutine stop_perf_collection()
        !! Stops the collection of performance data.
        !! Uses Intel Advisor annotations for that. If library was not found,
        !! this subroutine will do nothing.
#ifdef ADV_ANNOTATE
        call annotate_disable_collection_push()
#endif
    end subroutine

    subroutine print_usage()
        !! Prints out the attended usage of the operator benchmark  executable
        write(logger_get_info_channel(), *) &
            "usage: benchmark_operators [-h] [-c] [-o OUTPUT_DIR] &
            &[-i PARAMETER_FILE] [-n TIMESTEP]"
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "optional arguments:"
        write(logger_get_info_channel(), *) &
            "-h                 show this help message and exit"
        write(logger_get_info_channel(), *) &
            "-c                 check the parameter file"
        write(logger_get_info_channel(), *) &
            "-o OUTPUT_DIR      specifies the directory for the program output"
        write(logger_get_info_channel(), *) &
            "-i PARAMETER_FILE  specifies the directory for the input &
            &parameter file. If not provided, searches for params_in.txt &
            &in the current directory"
        write(logger_get_info_channel(), *) &
            "-n TIMESTEP        specifies the number of timesteps (default: 20)"

        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) &
            "caution: -cxx, -acc, -ompx only affect the supported operators,"
        write(logger_get_info_channel(), *) &
            "         otherwise the operators run with the pure Fortran &
            &version."
    end subroutine

    subroutine check_parameters(parameter_file)
        !! Checks if the parameter file is correct
        character(len=*), intent(in) :: parameter_file
        !! Parameter file name

        ! Check namelists in the parameter file
        write(logger_get_info_channel(), *) "Checking ", parameter_file
        call set_parameter_file(parameter_file)
        call read_parameter_file()
        write(logger_get_info_channel(), *) "params_parallelization: passed!"
    end subroutine

    subroutine benchmark_set_uniform()
        !! Benchmarks the op_set_uniform_cpu_t operator
        real(kind=GP), allocatable, target, dimension(:,:,:,:,:) :: f
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_ptr

        allocate(f(n_rz, n_phi, n_vp, n_mu, n_sp))
        f_ptr => f

        call op_set_uniform%reset_perf_counter()

        call start_perf_collection()
        do t = 1, n_t
            call op_set_uniform%apply(f, 2.0_GP)
            write(logger_get_debug_channel(), *) f_ptr(1,1,1,1,1)
        enddo
        call stop_perf_collection()

        call op_set_uniform%print_performance_summary()
    end subroutine

    subroutine benchmark_axpy()
        !! Benchmarks the op_axpy_cpu_t operator
        use op_axpy_m, only: op_axpy_cpu_t

        type(op_axpy_cpu_t) :: op_axpy
        real(kind=GP), allocatable, target, dimension(:,:,:,:,:) :: a_5d, b_5d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: a_ptr_5d, &
                                                                    b_ptr_5d
        real(kind=GP) :: alpha

        allocate(a_5d(n_rz, n_phi, n_vp, n_mu, n_sp))
        allocate(b_5d(n_rz, n_phi, n_vp, n_mu, n_sp))

        a_ptr_5d => a_5d
        b_ptr_5d => b_5d

        ! first touch initialize
        call op_set_uniform%apply(a_ptr_5d, 1.0_GP)
        call op_set_uniform%apply(b_ptr_5d, 1.0_GP)
        alpha = 1.0_GP

        call op_axpy%initialize()

        call start_perf_collection()
        do t = 1, n_t
            call op_axpy%apply(a_ptr_5d, alpha, b_ptr_5d)
            write(logger_get_debug_channel(), *) a_ptr_5d(1,1,1,1,1)
        enddo
        call stop_perf_collection()

        call op_axpy%print_performance_summary()
    end subroutine

    subroutine benchmark_copy()
        !! Benchmarks the op_copy_cpu_5d_t operator
        use op_copy_m, only: op_copy_cpu_t

        type(op_copy_cpu_t) :: op_copy
        real(kind=GP), allocatable, target, dimension(:,:,:,:,:) :: a_5d, b_5d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: a_ptr_5d, &
                                                                    b_ptr_5d

        allocate(a_5d(n_rz, n_phi, n_vp, n_mu, n_sp))
        allocate(b_5d(n_rz, n_phi, n_vp, n_mu, n_sp))

        a_ptr_5d => a_5d
        b_ptr_5d => b_5d

        ! First touch initialize
        call op_set_uniform%apply(a_ptr_5d, 1.0_GP)
        call op_set_uniform%apply(b_ptr_5d, 1.0_GP)

        call op_copy%initialize()

        call start_perf_collection()
        do t = 1, n_t
            call op_copy%apply(a_ptr_5d, b_ptr_5d)
            write(logger_get_debug_channel(), *) a_ptr_5d(1,1,1,1,1)
        enddo
        call stop_perf_collection()

        call op_copy%print_performance_summary()
    end subroutine

    subroutine benchmark_lin_comb()
        !! Benchmarks the op_lin_comb_cpu_t operator
        use op_lin_comb_m, only: op_lin_comb_cpu_t

        type(op_lin_comb_cpu_t) :: op_lin_comb
        real(kind=GP), allocatable, target, dimension(:,:,:,:,:) :: a_5d, &
                                                                    b_5d, c_5d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            a_ptr_5d, b_ptr_5d, c_ptr_5d
        real(kind=GP) :: alpha, beta

        allocate(a_5d(n_rz, n_phi, n_vp, n_mu, n_sp))
        allocate(b_5d(n_rz, n_phi, n_vp, n_mu, n_sp))
        allocate(c_5d(n_rz, n_phi, n_vp, n_mu, n_sp))

        a_ptr_5d => a_5d
        b_ptr_5d => b_5d
        c_ptr_5d => c_5d

        ! First touch initialize
        call op_set_uniform%apply(a_ptr_5d, 1.0_GP)
        call op_set_uniform%apply(b_ptr_5d, 1.0_GP)
        call op_set_uniform%apply(c_ptr_5d, 1.0_GP)
        alpha = 1.0_GP
        beta = 1.0_GP

        call op_lin_comb%initialize()

        call start_perf_collection()
        do t = 1, n_t
            call op_lin_comb%apply(a_ptr_5d, alpha, b_ptr_5d, beta, &
                                   c_ptr_5d)
            write(logger_get_debug_channel(), *) a_ptr_5d(1,1,1,1,1)
        enddo
        call stop_perf_collection()

        call op_lin_comb%print_performance_summary()
    end subroutine

    subroutine benchmark_mom_maxwells_eq()
        !! Benchmarks the op_mom_maxwells_eq_t operator
        use bsg_operators_m, only: bsg_operators_t
        use op_mom_maxwells_eq_m, only: op_mom_maxwells_eq_base_t, &
                                        op_mom_maxwells_eq_cpu_t, &
                                        op_mom_maxwells_eq_vspec_cpu_t
        class(op_mom_maxwells_eq_base_t), allocatable :: op_mom_maxwells_eq
        class(bsg_operators_t), allocatable :: bsg_op
        class(data_array_5d_t), allocatable :: da_f
        class(data_array_2d_t), allocatable :: da_co_qn_eq, da_b_qn_eq, &
                                               da_b_amps_law, da_b_bpar_eq
        real(kind=GP), contiguous, pointer, dimension(:,:) :: b_qn_eq

        allocate(da_f, da_co_qn_eq, da_b_qn_eq, da_b_bpar_eq, da_b_amps_law)
        call da_f%initialize([1,1,1,1,1], [n_rz, n_phi, n_vp, n_mu, n_sp])
        call da_b_qn_eq%initialize([1,1], [n_rz, n_phi])
        call da_b_amps_law%initialize(mold=da_b_qn_eq)
        call da_co_qn_eq%initialize(mold=da_b_qn_eq)
        call da_b_bpar_eq%initialize(mold=da_b_qn_eq)
        b_qn_eq => da_b_qn_eq%get_pointer()

        if(get_use_vspectral()) then
            allocate(op_mom_maxwells_eq_vspec_cpu_t :: op_mom_maxwells_eq)
         else
            allocate(op_mom_maxwells_eq_cpu_t :: op_mom_maxwells_eq)

            allocate(bsg_operators_t :: bsg_op)
            if (get_use_bsg()) then
                call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                       get_bsg_interp_order())
            else
                call bsg_op%initialize(mesh, 1)
            end if

        endif

        select type(op => op_mom_maxwells_eq)
            type is(op_mom_maxwells_eq_vspec_cpu_t)
                call op%initialize(dcomm_handler, mesh)
            type is(op_mom_maxwells_eq_cpu_t)
                call op%initialize(dcomm_handler, mesh, bsg_op)
        end select

        call start_perf_collection()
        do t = 1, n_t
            call op_mom_maxwells_eq%apply(da_f, da_co_qn_eq, &
                                          da_b_qn_eq, da_b_amps_law, &
                                          da_b_bpar_eq)
            write(logger_get_debug_channel(), *) b_qn_eq(1, 1)
        enddo
        call stop_perf_collection()

        call op_mom_maxwells_eq%print_performance_summary()

        deallocate(da_f, da_co_qn_eq, da_b_qn_eq, &
                   da_b_amps_law, da_b_bpar_eq)
        if(allocated(bsg_op)) deallocate(bsg_op)
    end subroutine

    subroutine benchmark_mom_0d()
        !! Benchmarks the op_diag_mom_0d_t operator
        use op_diag_mom_0d_m, only: op_diag_mom_0d_base_t, &
                                    op_diag_mom_0d_t, &
                                    op_diag_mom_0d_cpu_t, &
                                    op_diag_mom_0d_vspec_t, &
                                    op_diag_mom_0d_vspec_cpu_t
        use bsg_operators_m, only: bsg_operators_t

        class(op_diag_mom_0d_base_t), allocatable :: op_diag_mom_0d
        class(data_array_5d_t), allocatable :: da_f
        class(data_array_2d_t), allocatable :: da_es_pot
        class(data_array_2d_t), allocatable :: da_mom_0d
        real(kind=GP), contiguous, pointer, dimension(:,:) :: mom_0d
        class(bsg_operators_t), allocatable :: bsg_op

        allocate(da_f, da_es_pot, da_mom_0d)
        call da_f%initialize([1,1,1,1,1], [n_rz, n_phi, n_vp, n_mu, n_sp])
        call da_es_pot%initialize([1,1], [n_rz, n_phi])
        call da_mom_0d%initialize([1,1], [6, n_sp])
        mom_0d => da_mom_0d%get_pointer()

        allocate(bsg_operators_t :: bsg_op)
        if (get_use_bsg()) then
            call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                   get_bsg_interp_order())
        else
            call bsg_op%initialize(mesh, 1)
        end if

        if(get_use_vspectral()) then
            allocate(op_diag_mom_0d_vspec_cpu_t :: op_diag_mom_0d)
            select type(op => op_diag_mom_0d)
                class is(op_diag_mom_0d_vspec_t)
                    call op%initialize(dcomm_handler, mesh)
            end select
        else
            allocate(op_diag_mom_0d_cpu_t :: op_diag_mom_0d)
            select type(op => op_diag_mom_0d)
                class is(op_diag_mom_0d_t)
                    call op%initialize(dcomm_handler, mesh, bsg_op)
            end select
        endif

        call start_perf_collection()
        do t = 1, n_t
            call op_diag_mom_0d%apply(da_f, da_es_pot, da_mom_0d)
            write(logger_get_debug_channel(), *) mom_0d(1, 1)
        enddo
        call stop_perf_collection()

        call op_diag_mom_0d%print_performance_summary()

        deallocate(da_f, da_es_pot, da_mom_0d)
    end subroutine

    subroutine benchmark_mom_2d()
        !! Benchmarks the op_diag_mom_2d_t operator
        use op_diag_mom_2d_m, only: op_diag_mom_2d_base_t, &
                                    op_diag_mom_2d_cpu_t, &
                                    op_diag_mom_2d_t, &
                                    op_diag_mom_2d_vspec_cpu_t, &
                                    op_diag_mom_2d_vspec_t
        use bsg_operators_m, only: bsg_operators_t

        class(op_diag_mom_2d_base_t), allocatable :: op_diag_mom_2d
        class(data_array_5d_t), allocatable :: da_f
        class(data_array_4d_t), allocatable :: da_mom_2d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: mom_2d
        class(bsg_operators_t), allocatable :: bsg_op

        allocate(da_f, da_mom_2d)
        call da_f%initialize([1,1,1,1,1], [n_rz, n_phi, n_vp, n_mu, n_sp])
        call da_mom_2d%initialize([1,1,1,1], [n_rz, n_phi, 8, n_sp])
        mom_2d => da_mom_2d%get_pointer()

        allocate(bsg_operators_t :: bsg_op)
        if (get_use_bsg()) then
            call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                   get_bsg_interp_order())
        else
            call bsg_op%initialize(mesh, 1)
        end if

        if(get_use_vspectral()) then
            allocate(op_diag_mom_2d_vspec_cpu_t :: op_diag_mom_2d)
            select type(op => op_diag_mom_2d)
                class is(op_diag_mom_2d_vspec_t)
                    call op%initialize(dcomm_handler, mesh)
            end select
        else
            allocate(op_diag_mom_2d_cpu_t :: op_diag_mom_2d)
            select type(op => op_diag_mom_2d)
                class is(op_diag_mom_2d_t)
                    call op%initialize(dcomm_handler, mesh, bsg_op)
            end select
        endif

        ! Run the benchmark
        ! Prerun the operator one because it allocates buffer upon first
        ! invocation
        call op_diag_mom_2d%apply(da_f, da_mom_2d)
        call op_diag_mom_2d%reset_perf_counter()

        call start_perf_collection()
        do t = 1, n_t
            call op_diag_mom_2d%apply(da_f, da_mom_2d)
            write(logger_get_debug_channel(), *) mom_2d(1, 1, 1, 1)
        enddo
        call stop_perf_collection()

        call op_diag_mom_2d%print_performance_summary()
        deallocate(da_f, da_mom_2d)
    end subroutine

    subroutine benchmark_rhs_vlasov_eq_static()
        !! Benchmarks the op_rhs_vlasov_eq_static_t operator
        use op_rhs_vlasov_eq_static_m, only: op_rhs_vlasov_eq_static_base_t, &
                                             op_rhs_vlasov_eq_static_cpu_t, &
                                             op_rhs_vlasov_eq_static_vspec_cpu_t
        use bsg_operators_m, only: bsg_operators_t
#ifdef ENABLE_GPU
        use op_rhs_vlasov_eq_static_m, only: op_rhs_vlasov_eq_static_gpu_t
#endif
        class(op_rhs_vlasov_eq_static_base_t), allocatable :: &
            op_rhs_vlasov_eq_static
        class(data_array_5d_t), allocatable :: da_f_in, da_f_out
        class(data_array_2d_t), allocatable :: da_phi_in, da_A_par_in, &
                                               da_B_par_in
        class(bsg_operators_t), allocatable, target :: bsg_op
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_out
        integer :: lb_stripped(5), ub_stripped(5), lb(5), ub(5), &
                   shp_stripped(5), shp_ghost(5)

        ! Setup data storage for correct ghost cell handling
        shp_stripped = [n_rz, n_phi, n_vp, n_mu, n_sp]
        shp_ghost  = [0, 2, 2, 0, 0]
        lb_stripped = [1, 1, 1, 1, 1]
        ub_stripped = shp_stripped
        lb = lb_stripped - [0, 2, 2, 0, 0]
        ub = ub_stripped + shp_ghost

        ! Allocate and first touch initialize distribution functions and
        ! the electromagnetic fields
        allocate(da_f_in, da_phi_in, da_A_par_in, da_B_par_in, da_f_out)
        call da_f_in%initialize(lb, ub, lb_stripped, ub_stripped, val=1.0_GP)
        call da_phi_in%initialize(lb(1:2), ub(1:2), val=1.0_GP)
        call da_A_par_in%initialize(lb(1:2), ub(1:2), val=1.0_GP)
        call da_B_par_in%initialize(lb(1:2), ub(1:2), val=1.0_GP)
        call da_f_out%initialize(lb, ub, val=1.0_GP)
        f_out => da_f_out%get_pointer()

        if(get_use_vspectral()) then
            allocate(op_rhs_vlasov_eq_static_vspec_cpu_t :: &
                     op_rhs_vlasov_eq_static)
        else
            if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
                allocate(op_rhs_vlasov_eq_static_gpu_t :: &
                         op_rhs_vlasov_eq_static)
#endif
            else
                allocate(op_rhs_vlasov_eq_static_cpu_t :: &
                         op_rhs_vlasov_eq_static)
            endif

            allocate(bsg_operators_t :: bsg_op)
            if (get_use_bsg()) then
                call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                       get_bsg_interp_order())
            else
                call bsg_op%initialize(mesh, 1)
            end if
        endif

        select type(op => op_rhs_vlasov_eq_static)
            type is(op_rhs_vlasov_eq_static_vspec_cpu_t)
                call op%initialize(mesh)
            type is(op_rhs_vlasov_eq_static_cpu_t)
                call op%initialize(mesh, bsg_op)
#ifdef ENABLE_GPU
            type is(op_rhs_vlasov_eq_static_gpu_t)
                call op%initialize(mesh, bsg_op)
#endif
        end select

        call start_perf_collection()
        do t = 1, n_t
            call op_rhs_vlasov_eq_static%apply(da_f_in, da_phi_in, &
                                               da_A_par_in, da_B_par_in, &
                                               da_f_out)
            write(logger_get_debug_channel(), *) f_out(1, 1, 1, 1, 1)
        enddo
        call stop_perf_collection()

        call op_rhs_vlasov_eq_static%print_performance_summary()

        deallocate(op_rhs_vlasov_eq_static)
        deallocate(da_f_in, da_phi_in, da_A_par_in, da_B_par_in, da_f_out)
        if(allocated(bsg_op)) deallocate(bsg_op)

    end subroutine

    subroutine benchmark_rhs_vlasov_eq_dynamic()
        !! Benchmarks the op_rhs_vlasov_eq_dynamic_t operator
        use bsg_operators_m, only: bsg_operators_t
        use op_rhs_vlasov_eq_dynamic_m, only: &
                                          op_rhs_vlasov_eq_dynamic_base_t, &
                                          op_rhs_vlasov_eq_dynamic_cpu_t, &
                                          op_rhs_vlasov_eq_dynamic_vspec_cpu_t
#ifdef ENABLE_GPU
        use op_rhs_vlasov_eq_dynamic_m, only: op_rhs_vlasov_eq_dynamic_gpu_t
#endif

        class(op_rhs_vlasov_eq_dynamic_base_t), allocatable :: &
            op_rhs_vlasov_eq_dynamic
        class(data_array_5d_t), allocatable :: da_f_in, da_f_out
        class(data_array_2d_t), allocatable :: da_E_par_in
        class(bsg_operators_t), allocatable :: bsg_op
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_out
        integer :: lb_stripped(5), ub_stripped(5), lb(5), ub(5), &
                   shp_stripped(5), shp_ghost(5)

        ! Setup data storage for correct ghost cell handling
        shp_stripped = [n_rz, n_phi, n_vp, n_mu, n_sp]
        shp_ghost  = [0, 2, 2, 0, 0]
        lb_stripped = [1, 1, 1, 1, 1]
        ub_stripped = shp_stripped
        lb = lb_stripped - [0, 2, 2, 0, 0]
        ub = ub_stripped + shp_ghost

        ! Allocate and first touch initialize distribution functions and
        ! the electromagnetic fields
        allocate(da_f_in, da_E_par_in, da_f_out)
        call da_f_in%initialize(lb, ub, lb_stripped, ub_stripped, val=1.0_GP)
        call da_E_par_in%initialize(lb(1:2), ub(1:2), val=1.0_GP)
        call da_f_out%initialize(lb, ub, val=1.0_GP)
        f_out => da_f_out%get_pointer()

        if(get_use_vspectral()) then
            allocate(op_rhs_vlasov_eq_dynamic_vspec_cpu_t :: &
                     op_rhs_vlasov_eq_dynamic)
        else
            if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
                allocate(op_rhs_vlasov_eq_dynamic_gpu_t :: &
                         op_rhs_vlasov_eq_dynamic)
#endif
            else
                allocate(op_rhs_vlasov_eq_dynamic_cpu_t :: &
                         op_rhs_vlasov_eq_dynamic)
            endif

            allocate(bsg_operators_t :: bsg_op)
            if (get_use_bsg()) then
                call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                       get_bsg_interp_order())
            else
                call bsg_op%initialize(mesh, 1)
            end if
        endif

        select type(op => op_rhs_vlasov_eq_dynamic)
            type is(op_rhs_vlasov_eq_dynamic_vspec_cpu_t)
                call op%initialize(mesh)
            type is(op_rhs_vlasov_eq_dynamic_cpu_t)
                call op%initialize(mesh, bsg_op)
#ifdef ENABLE_GPU
            type is(op_rhs_vlasov_eq_dynamic_gpu_t)
                call op%initialize(mesh, bsg_op)
#endif
        end select

        call start_perf_collection()
        do t = 1, n_t
            call op_rhs_vlasov_eq_dynamic%apply(da_f_in, da_E_par_in, da_f_out)
            write(logger_get_debug_channel(), *) f_out(1, 1, 1, 1, 1)
        enddo
        call stop_perf_collection()

        call op_rhs_vlasov_eq_dynamic%print_performance_summary()

        deallocate(op_rhs_vlasov_eq_dynamic)
        deallocate(da_f_in, da_E_par_in, da_f_out)
        if(allocated(bsg_op)) deallocate(bsg_op)
    end subroutine

    subroutine benchmark_mom_coll(mom_coll_type)
        !! Benchmarks the op_mom_coll_t operator
        use op_mom_coll_m, only: op_mom_coll_base_t, &
                                 op_mom_coll_cpu_t, &
                                 op_mom_coll_vspec_cpu_t

        character(len=*), intent(in) :: mom_coll_type

        class(op_mom_coll_base_t), allocatable :: op_mom_coll
        class(data_array_5d_t), allocatable :: da_f_in
        class(data_array_4d_t), allocatable :: da_moments
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments

        allocate(da_f_in)
        allocate(da_moments)
        call da_f_in%initialize([1,1,1,1,1], [n_rz, n_phi, n_vp, n_mu, n_sp], &
                                val=1.0_GP)
        call da_moments%initialize([1,1,1,1], [n_rz, n_phi, 3, n_sp])
        moments => da_moments%get_pointer()

        if(mom_coll_type == "op_mom_coll_cpu_t") then
            allocate(op_mom_coll_cpu_t :: op_mom_coll)
        else if(mom_coll_type == "op_mom_coll_vspec_cpu_t") then
            allocate(op_mom_coll_vspec_cpu_t :: op_mom_coll)
        else
            call handle_error("op_mom_coll type not implemented!", &
                              GENEX_ERR_BENCHMARKS, __LINE__, __FILE__)
        endif

        call op_mom_coll%initialize(dcomm_handler, mesh)

        call start_perf_collection()
        do t = 1, n_t
            call op_mom_coll%apply(da_f_in, da_moments)
            write(logger_get_debug_channel(), *) moments(1, 1, 1, 1)
        enddo
        call stop_perf_collection()

        call op_mom_coll%print_performance_summary()

        deallocate(da_moments, da_f_in)
    end subroutine

    subroutine benchmark_coll(coll_type)
        !! Benchmarks collision operators
        use op_coll_m, only: op_coll_base_t, &
                             op_coll_t, &
                             op_coll_vspec_t, &
                             op_coll_bgk_cpu_t, &
                             op_coll_lbd_cpu_t, &
                             op_coll_lbd_vspec_cpu_t, &
                             op_coll_lorentz_cpu_t
        use op_mom_coll_m, only: op_mom_coll_base_t, &
                                 op_mom_coll_cpu_t, &
                                 op_mom_coll_vspec_cpu_t

        character(len=*), intent(in) :: coll_type

        class(op_coll_base_t), allocatable :: op_coll
        class(op_mom_coll_base_t), allocatable :: op_mom_coll
        class(data_array_5d_t), allocatable :: da_f_in, da_f_out
        class(data_array_4d_t), allocatable :: da_moments
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_out
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped, &
                                 shp_stripped, shp_ghost

        ! Setup data storage for correct ghost cell handling
        shp_stripped = [n_rz, n_phi, n_vp, n_mu, n_sp]
        shp_ghost  = [0, 2, 2, 1, 0]
        lb_stripped = [1, 1, 1, 1, 1]
        ub_stripped = shp_stripped
        lb = lb_stripped - [0, 2, 2, 1, 0]
        ub = ub_stripped + shp_ghost

        allocate(da_f_in, da_f_out)
        allocate(da_moments)
        call da_f_in%initialize(lb, ub, lb_stripped, ub_stripped, val=1.0_GP)
        call da_f_out%initialize(lb, ub, lb_stripped, ub_stripped, val=1.0_GP)
        call da_moments%initialize([lb_stripped(1), lb_stripped(2), 1, 1], &
                                   [ub_stripped(1), ub_stripped(2), 3, n_sp])
        f_out => da_f_out%get_pointer()

        if(get_use_vspectral()) then
            allocate(op_mom_coll_vspec_cpu_t :: op_mom_coll)
        else
            allocate(op_mom_coll_cpu_t :: op_mom_coll)
        endif

        if(coll_type == "bgk" .and. .not. get_use_vspectral()) then
            allocate(op_coll_bgk_cpu_t :: op_coll)
        else if(coll_type == "lbd" .and. .not. get_use_vspectral()) then
            allocate(op_coll_lbd_cpu_t :: op_coll)
        else if(coll_type == "lbd" .and. get_use_vspectral()) then
            allocate(op_coll_lbd_vspec_cpu_t :: op_coll)
        else if(coll_type == "lorentz" &
                .and. .not. get_use_vspectral()) then
            allocate(op_coll_lorentz_cpu_t :: op_coll)
        else
            call handle_error("Collision type not implemented!", &
                              GENEX_ERR_BENCHMARKS, __LINE__, __FILE__)
        endif

        select type(op_coll)
            class is(op_coll_t)
                call op_coll%initialize(dcomm_handler, mesh)
            class is(op_coll_vspec_t)
                call op_coll%initialize(dcomm_handler, mesh)
        end select

        call op_mom_coll%initialize(dcomm_handler, mesh)
        call op_mom_coll%apply(da_f_in, da_moments)

        call start_perf_collection()
        do t = 1, n_t
            call op_coll%apply(da_f_in, da_moments, da_f_out)
            write(logger_get_debug_channel(), *) f_out(1, 1, 1, 1, 1)
        enddo
        call stop_perf_collection()

        call op_coll%print_performance_summary()

        deallocate(da_moments, da_f_in, da_f_out)
    end subroutine

end program benchmark_operators
