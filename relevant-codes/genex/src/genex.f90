program genex
    !! Main genex program
    use mpi
    use genex_build_info_m, only: git_hash, git_date
    use genex_fortran_env_m, only: DP, GP
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_MAIN, GENEX_SUCCESS, &
                                    GENEX_WRN_FILE_NOT_EXIST
    use logger_m, only: logger_get_error_channel, logger_set_debug_channel, &
                        logger_get_debug_channel, logger_get_info_channel, &
                        logger_set_info_channel, logger_welcome
    use profiler_m, only: profiler_start, profiler_stop, profiler_set_path, &
                          profiler_print, profiler_set_err_mode, &
                          PROFILER_ERR_FATAL
    use type_converters_m, only: string
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use timestep_m, only: timestep_t, rk4_t, euler_t, strang_splitting_t, &
                          pirock_t
    use params_m, only: set_parameter_file, read_parameter_file, &
                        write_parameter_file
    use error_params_m, only: set_print_info
    use params_parallelization_m, only: get_n_procs_phi, get_n_procs_vp, &
                                        get_n_procs_mu, get_n_procs_sp
    use params_time_loop_m, only: get_dt, get_n_timesteps, &
                                  get_wrap_up_time, &
                                  get_steps_diagnostics, &
                                  get_steps_checkpoint, &
                                  get_time_stepping_scheme, &
                                  get_start_from_checkpoint, &
                                  get_run_mms
    use file_handling_m, only: file_exists, delete_file_if_exists
    use params_gpu_offload_m, only: get_use_gpu_offload, &
                                    get_gpu_offload_backend, &
                                    get_parallax_gpu_offload_backend, &
                                    check_gpu_functionalities, &
                                    GPU_OFFLOAD_CPU, &
                                    GPU_OFFLOAD_ACC, &
                                    GPU_OFFLOAD_OMPX, &
                                    PARALLAX_BACKEND_GPU, &
                                    PARALLAX_BACKEND_RLU, &
                                    PARALLAX_BACKEND_AMGX, &
                                    PARALLAX_BACKEND_HYPRE
    use params_diagnostics_m, only: get_diagnose_tpc
    use devtools_m, only: initialize_devtools, finalize_devtools

    ! From PARALLAX
    use screen_io_m, only: set_parallax_stdout => set_stdout, &
                           set_parallax_stderr => set_stderr
    use device_handling_m, only: impose_default_device_affinity

    implicit none

    integer :: argc, i, ierr, debug_channel
    character(len=256), allocatable, dimension(:) :: argv
    character(len=256) :: out_dir, parameter_file
    character(len=64) :: runtime_version_string
    integer :: runtime_limit = huge(0)
    logical :: parameter_check

    ! Process command line arguments
    argc = command_argument_count()
    if(argc < 1 .or. argc > 7) then
        call print_usage()
        stop
    endif

    ! Retrieve all arguments
    allocate(argv(argc))
    do i = 1, argc
        call get_command_argument(i, argv(i))
    enddo

    ! Get the parameter file
    if(argv(argc)(1:1) == "-") then
        ! The parameter file may not be an optional variable
        call print_usage()
        stop
    endif
    parameter_file = argv(argc)

    ! Parse the optional arguments
    out_dir = ""
    parameter_check = .false.
    i = 1
    do while(i <= argc - 1)
        select case(argv(i))
        case("-c")
            parameter_check = .true.
        case("-o")
            if(i > argc - 2) then
                ! The output directory has to be specified after -o
                call print_usage()
                stop
            endif
            out_dir = argv(i + 1)
            if(out_dir(1:1) == "-") then
                ! The output directory may not be an optional variable
                call print_usage()
                stop
            endif
            i = i + 1
        case("-t")
            if(i > argc - 2) then
                ! The job runtime limit has to be specified after -t
                call print_usage()
                stop
            endif
            i = i + 1
            read (argv(i), "(I10)") runtime_limit
        end select
        i = i + 1
    end do

    deallocate(argv)

    ! Set debug log
    open(newunit=debug_channel, file=trim(out_dir)//'debug_log.txt', &
         status='replace', action='write', iostat=ierr)

    if(ierr /= 0) then
        call handle_error("Could not create debug log file!", &
                          GENEX_ERR_MAIN, __LINE__, __FILE__)
    endif

    ! Set parameter I/O info output
    call set_print_info(.true.)

    if(parameter_check) then
        call check_parameters(trim(parameter_file))
    else
        ! NOTE: Only redirect debug output for a simulation to the debug file.
        !       For the parameter check we want the output directly to stdout.
        call logger_set_debug_channel(debug_channel)
        ! Redirect PARALLAX output to the debug channel
        call set_parallax_stdout(debug_channel)

        call start_simulation(trim(parameter_file), trim(out_dir))
    endif

    close(debug_channel)

    ! Redirect PARALLAX error output to the error channel
    call set_parallax_stderr(logger_get_error_channel())
contains

    subroutine print_usage()
        !! Prints the attended usage of the genex executable on screen
        write(logger_get_info_channel(), *) &
            "usage: genex [-c] [-o OUTPUT_DIR] [-t RUNTIME_LIMIT] &
             PARAMETER_FILE"
        write(logger_get_info_channel(), *) ""
        write(logger_get_info_channel(), *) "optional arguments:"
        write(logger_get_info_channel(), *) &
            "-c               check the parameter file"
        write(logger_get_info_channel(), *) &
            "-o OUTPUT_DIR    specifies the directory for the program output"
        write(logger_get_info_channel(), *) &
            "-t RUNTIME_LIMIT specifies the job runtime limit in seconds"
    end subroutine

    subroutine check_parameters(parameter_file)
        !! Checks if the parameter file is correct
        character(len=*), intent(in) :: parameter_file
        !! Parameter file name

        call set_parameter_file(parameter_file)
        call read_parameter_file()

        ! If not terminated by error, check was successful
        write(logger_get_info_channel(), *) "Check passed!"
    end subroutine

    subroutine print_input_summary()
      write(logger_get_info_channel(), "(2A)") &
           "Output is written to ", out_dir
    end subroutine

    subroutine start_simulation(parameter_file, out_dir)
        !! Starts the simulation
        character(len=*), intent(in) :: parameter_file
        !! Parameter file
        character(len=*), intent(in) :: out_dir
        !! Output directory

        type(dcomm_handler_t), allocatable, target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr

        integer :: ierr, thread_level
        ! Error code and MPI thread level
        integer :: dim_permut(5)

        real(kind=DP) :: start_wtime
        ! Wall clock time of the simulation start

        ! Save the parameter file name in params_m in order to pass it
        ! to PARALLAX
        call set_parameter_file(parameter_file)
        call read_parameter_file()

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

        ! Initialize MPI with or without OpenMP
#ifdef _OPENMP
        call MPI_Init_thread(MPI_THREAD_FUNNELED, thread_level, ierr)
        if(thread_level /= MPI_THREAD_FUNNELED) then
            call handle_error("Hybrid MPI + OpenMP not supported!", &
                              GENEX_ERR_MAIN, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                "Thread level provided was:", [thread_level]))
        endif
#else
        call MPI_Init(ierr)
#endif

        ! Check if GPUs are found and can be used as intended
        call check_gpu_functionalities()

        ! Initialize the developer tools if any are enabled
        call initialize_devtools(MPI_COMM_WORLD, out_dir)

        ! Get wall clock time to start the runtime tracking
        start_wtime = MPI_wtime()

        dim_permut = [1, 2, 3, 4, 5]
        allocate(dcomm_handler)
        call dcomm_handler%initialize(MPI_COMM_WORLD, get_n_procs_phi(), &
                                     get_n_procs_vp(), get_n_procs_mu(), &
                                     get_n_procs_sp(), dim_permut)
        dcomm_handler_ptr => dcomm_handler

        ! Welcome the user
        if(dcomm_handler%is_master()) then

            if(.not. get_run_mms()) then
                ! Only print this for non-MMS runs to not mess up the output
                ! that is essential for MMS test
                call logger_welcome()
                write(logger_get_info_channel(), *) ""
            endif

            write(logger_get_info_channel(), *) "Welcome to GENE-X!"
            write(logger_get_info_channel(), *) ""
            write(logger_get_info_channel(), *) "   You are running commit ", &
                                                git_hash
            write(logger_get_info_channel(), *) "   From ", git_date
            write(logger_get_info_channel(), *) ""
        endif

        if(dcomm_handler%is_master()) then
            if (.not.get_run_mms()) then
                call print_input_summary()
            endif
            call write_parameter_file(out_dir)
        endif

        flush(logger_get_info_channel())
        flush(logger_get_debug_channel())
        flush(logger_get_error_channel())

        ! Set GPU affinity for PARALLAX GPU feature in the case where
        ! GENE-X kernels run on CPU
        if(get_gpu_offload_backend() == GPU_OFFLOAD_CPU) then
            select case(get_parallax_gpu_offload_backend())
                case(PARALLAX_BACKEND_GPU)
                    call impose_default_device_affinity('MGMRES_GPU')
                case(PARALLAX_BACKEND_RLU)
                    call impose_default_device_affinity('ROCALUTION')
                case(PARALLAX_BACKEND_AMGX)
                    call impose_default_device_affinity('AMGX')
                case(PARALLAX_BACKEND_HYPRE)
                    call impose_default_device_affinity('HYPRE')
            end select
        endif

        ! Run the time loop
        call time_loop(dcomm_handler_ptr, out_dir, start_wtime)

        deallocate(dcomm_handler)

        ! Finalize the developer tools if any is enabled
        call finalize_devtools(MPI_COMM_WORLD)

        ! Finalize MPI
        call MPI_Finalize(ierr)
    end subroutine

    subroutine time_loop(dcomm_handler, out_dir, start_wtime)
        !! Runs the time loop
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        !! MPI communication handler
        character(len=*), intent(in) :: out_dir
        !! Output directory
        real(kind=DP), intent(in) :: start_wtime
        !! Wall clock time at the start of the simulation

        type(mesh_5d_t), target, allocatable :: mesh
        type(mesh_5d_t), pointer :: mesh_ptr
        class(timestep_t), target, allocatable :: timestep

        integer :: t, n_timesteps
        ! Current time step and number of time steps
        integer :: steps_diagnostics, steps_checkpoint
        ! Number of steps between saving diagnostics and checkpoint
        integer :: count_diagnostics, count_checkpoint
        ! Counter of steps between saving diagnostics and checkpoint
        integer :: wrap_up_time
        ! Time reserved for writing files before runtime is over
        integer :: ierr
        ! Error code
        integer :: ierr_finalize
        ! Error code used in the finalization of the run to not overwrite errors
        ! thrown inside the program
        logical :: start_from_checkpoint, run_mms, read_mesh_from_file
        ! Control variables for restart, mesh reading, and MMS runs
        logical :: diagnose_tpc
        ! Copy of the TPC condition to use it in several calls
        character(len=:), allocatable :: mesh_filename
        ! Name of mesh file to read from or write to
        character(len=:), allocatable :: error_message
        ! Error message to be displayed in an error event
        character(len=:), allocatable :: time_stepping_scheme
        ! Time stepping scheme that should be used for the simulation
        real(kind=GP) :: dt, calc_t_max
        ! Timestep and maximum simulation time
        real(kind=DP) :: current_wtime
        ! Current MPI wall clock time used for time measurements
        real(kind=GP) :: first_wtime
        ! Wall clock time after the first time step
        real(kind=GP) :: last_wtime, last_diagnostics_wtime
        ! Wall clock time at last time step and last execution of diagnostics
        real(kind=GP) :: time_per_step, max_time_per_step, min_time_per_step
        ! Time passed at the last time step and global max and min
        real(kind=GP) :: avg_time_per_step
        ! Time passed per step, averaged between executing diagnostics

        ierr = GENEX_SUCCESS

        ! We set the profiler errors to fatal to avoid having to handle
        ! errors on each call of the profiler
        call profiler_set_err_mode(PROFILER_ERR_FATAL)
        call profiler_start("genex", ierr, set_path=.true.)
        call profiler_start("initialize", ierr, set_path=.true.)

        ! Allocate and initialize the mesh
        call profiler_start("initialize_mesh", ierr, set_path=.true.)

        start_from_checkpoint = get_start_from_checkpoint()
        mesh_filename = trim(out_dir) // "mesh.nc"

        if(start_from_checkpoint) then
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
        else
            read_mesh_from_file = .false.
        endif

        allocate(mesh)
        call mesh%initialize(dcomm_handler, &
                             read_mesh_from_file=read_mesh_from_file, &
                             mesh_filename=mesh_filename)
        mesh_ptr => mesh

        call profiler_stop("initialize_mesh", ierr, set_path=.true.)

        if(.not. dcomm_handler%is_initialized()) then
            error_message = "MPI domain decomposition was not properly &
                            &initialized by mesh!"
            goto 300
        endif

        call dcomm_handler%print_mesh_info()

        ! Allocate and initialize the timestep
        dt                    = get_dt()
        n_timesteps           = get_n_timesteps()
        wrap_up_time          = get_wrap_up_time()
        steps_diagnostics     = get_steps_diagnostics()
        steps_checkpoint      = get_steps_checkpoint()
        time_stepping_scheme  = get_time_stepping_scheme()
        run_mms               = get_run_mms()
        ! TODO: Estimate n_timesteps with the CFL criterion here in the future
        calc_t_max  = n_timesteps * dt
        n_timesteps = min(n_timesteps, int(calc_t_max / dt))

        ! Select the time stepping scheme
        select case(time_stepping_scheme)
        case("rk4")
            allocate(rk4_t :: timestep)
        case("euler")
            allocate(euler_t :: timestep)
        case("strang_splitting")
            allocate(strang_splitting_t :: timestep)
        case("pirock")
            allocate(pirock_t :: timestep)
        case default
            ierr = GENEX_ERR_MAIN
            error_message = "Selected time stepping scheme is not supported!"
            goto 200
        end select

        if(dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) "Initialize time integrator &
                                                &with ", runtime_version_string
        endif

        call profiler_start("initialize_timestep", ierr, set_path=.true.)
        call timestep%initialize(dcomm_handler, mesh_ptr, dt, out_dir, &
                                 start_from_checkpoint, run_mms)
        call profiler_stop("initialize_timestep", ierr, set_path=.true.)

        call profiler_stop("initialize", ierr, set_path=.true.)

        ! NOTE: The following output needs to be done before the initial
        !       file I/O, otherwise the output table of the MMS test will
        !       be out of order and tests will fail. To be consistent with
        !       the output message, we also stop the profiling of initialize-

        if(dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) "Initialization finished"
            write(logger_get_info_channel(), *) ""
            write(logger_get_info_channel(), *) "Time integration"
            write(logger_get_info_channel(), *) ""
        endif

        ! Initialize controls counters for the diagnostics and the checkpoint
        ! save
        count_diagnostics = 1
        count_checkpoint  = 1

        ! Check if TPC is activated for the run
        diagnose_tpc = get_diagnose_tpc()

        ! Save an initial checkpoint
        ! NOTE: The initialization of the timestep will modify the
        !       start_from_checkpoint to be false if no checkpoint was found.
        !       In that case the following block will also be executed.
        if(.not. start_from_checkpoint) then
            call profiler_start("file_io", ierr, set_path=.true.)
            call timestep%diagnose_all(ierr)
            call timestep%save_checkpoint(ierr)
            call timestep%create_part
            call profiler_stop("file_io", ierr, set_path=.true.)
        endif

        ! Initialize time tracking variables
        current_wtime          = MPI_wtime()
        first_wtime            = current_wtime
        last_wtime             = current_wtime
        last_diagnostics_wtime = current_wtime
        min_time_per_step      = huge(0.0_GP)
        max_time_per_step      = 0.0_GP

        ! Time loop
        do t = 1, n_timesteps
            if(file_exists('stop')) then
                if(dcomm_handler%is_master()) then
                    write(logger_get_info_channel(), *) &
                        "Terminating time integration due to stop file"
                endif
                ! Terminate the program and write diagnostics on the final
                ! state
                goto 100
            endif

            ! Get current wall clock time for runtime tracker
            current_wtime = MPI_wtime()

            if(current_wtime - start_wtime >= runtime_limit - wrap_up_time) then
                if(dcomm_handler%is_master()) then
                    write(logger_get_info_channel(), *) &
                        "Terminating time integration due to job runtime limit"
                endif
                ! Terminate the program and write diagnostics on the final
                ! state
                goto 100
            endif

            ! Calculate timestep
            call profiler_start("timestep", ierr, set_path=.true.)

            call timestep%step(ierr)
            if(ierr /= GENEX_SUCCESS) then
                error_message = "Terminating time integration due to &
                                &error in timestep!"
                ! Terminate the program without writing diagnostics on the
                ! final state
                goto 200
            endif

            call profiler_stop("timestep", ierr, set_path=.true.)

            call profiler_start("file_io", ierr, set_path=.true.)

            ! Calculate time per step
            ! NOTE: We skip the first timestep due to initializations that
            !       disturb the average time calculated.
            if (t > 1) then
                time_per_step = (current_wtime - last_wtime)
                max_time_per_step = max(time_per_step, max_time_per_step)
                min_time_per_step = min(time_per_step, min_time_per_step)
            endif
            last_wtime = current_wtime
            ! Set wtime after the first time step (after init is done)
            if(t == 2) first_wtime = MPI_wtime()

            ! Execute diagnostics
            if(count_diagnostics == steps_diagnostics) then

                call timestep%diagnose_all(ierr)

                count_diagnostics = 0

                avg_time_per_step = current_wtime - last_diagnostics_wtime
                if(t == steps_diagnostics) then
                    ! NOTE: On first average calculation we need to subtract
                    !       one from the number of steps because we start with
                    !       count_diagnostics = 1.
                    !       Further we need to subtract one because we skip
                    !       measurement in the first timestep.
                    avg_time_per_step = avg_time_per_step &
                                      / (steps_diagnostics - 2)
                else
                    avg_time_per_step = avg_time_per_step / steps_diagnostics
                endif
                last_diagnostics_wtime = current_wtime

                if(dcomm_handler%is_master() .and. .not. run_mms) then
                    write(logger_get_info_channel(), *) &
                        "Writing diagnostics at step "//string(t)&
                        //" / "//string(n_timesteps)

                    write(logger_get_info_channel(), *) &
                        "Average execution time per timestep over the last "&
                        //string(steps_diagnostics)//" steps: " &
                        //string(avg_time_per_step, 2)//" s"
                endif

            endif

            ! Save checkpoint
            if(count_checkpoint == steps_checkpoint) then
                call timestep%save_checkpoint(ierr)
                count_checkpoint = 0

                if(dcomm_handler%is_master() .and. .not. run_mms) then
                    write(logger_get_info_channel(), *) &
                        "Writing checkpoint at step  "//string(t)&
                        //" / "//string(n_timesteps)
                    write(logger_get_info_channel(), *) ""
                endif

                ! Create a new part for every checkpoint written. This is done
                ! to achieve maximum fail safety. If the simulation terminates
                ! due to unforeseen reasons the latest part directory, with
                ! possibly corrupt diagnostic files, can be easily removed and
                ! the simulation restarted from the last available checkpoint.
                if(t /= n_timesteps) then
                    call timestep%create_part()
                endif
            endif

            call profiler_stop("file_io", ierr, set_path=.true.)

            count_diagnostics = count_diagnostics + 1
            count_checkpoint  = count_checkpoint  + 1

            flush(logger_get_info_channel())
            flush(logger_get_debug_channel())
            flush(logger_get_error_channel())
        enddo

        if(dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) "Finalize time integration"
            write(logger_get_info_channel(), *) ""
        endif

        ! Execute the diagnostics and checkpoint on the final state
100     if(count_diagnostics /= 1) then
            call profiler_start("file_io", ierr, set_path=.true.)
            call timestep%diagnose_all(ierr)
            call profiler_stop("file_io", ierr, set_path=.true.)
        endif
        if(count_checkpoint /= 1) then
            call profiler_start("file_io", ierr, set_path=.true.)
            call timestep%save_checkpoint(ierr)
            call profiler_stop("file_io", ierr, set_path=.true.)
        endif

        ! Clean up and terminate the program
        ! Wait for all MPI processes to exit the time loop before we delete
        ! the stop file
200     call mpi_barrier(dcomm_handler%get_comm_cart(), ierr_finalize)
        if(dcomm_handler%is_master()) then
            call delete_file_if_exists('stop')
        endif

        if(allocated(timestep)) deallocate(timestep)
        if(allocated(mesh)) deallocate(mesh)

        if(dcomm_handler%is_master()) then
            avg_time_per_step = (current_wtime - start_wtime) / t

            write(logger_get_info_channel(), *) ""
            write(logger_get_info_channel(), *) &
                "Summary of execution time per timestep"
            write(logger_get_info_channel(), *) &
                "Average: "//string(avg_time_per_step, 2)//" s"
            write(logger_get_info_channel(), *) &
                "Maximum: "//string(max_time_per_step, 2)//" s"
            write(logger_get_info_channel(), *) &
                "Minimum: "//string(min_time_per_step, 2)//" s"
            write(logger_get_info_channel(), *) ""

            write(logger_get_info_channel(), *) "Profiler analysis"
            write(logger_get_info_channel(), *) ""
        endif

        call profiler_stop("genex", ierr_finalize)
        call profiler_print(dcomm_handler, logger_get_info_channel(), "genex", &
                            ierr_finalize, discard_first_call = .true.)

        ! Terminate with (error) message
300     if (ierr == GENEX_SUCCESS) then
            if(dcomm_handler%is_master()) then
                write(logger_get_info_channel(), *) ""
                write(logger_get_info_channel(), *) "Simulation finished"
                write(logger_get_info_channel(), *) ""
            endif
        else
            if(dcomm_handler%is_master()) then
                call handle_error(error_message, ierr, __LINE__, __FILE__)
            endif
        endif
    end subroutine

end program genex
