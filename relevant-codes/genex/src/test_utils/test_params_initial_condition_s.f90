submodule (test_params_m) test_params_initial_condition_s
    ! Submodule that contains helpers for unit tests to initialize
    ! initial condition with non-default parameters.
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_initial_condition(comm, rank, &
                                            initial_perturbation)
        !! Initializes initial condition with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        character(len=*), optional, intent(in) :: initial_perturbation

        character(len=:), allocatable :: initial_perturbation_local

        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        initial_perturbation_local = get_initial_perturbation()

        if(present(initial_perturbation)) &
                             initial_perturbation_local = initial_perturbation

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", &
                 iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_initial_condition")
            call write_nml(iunit, "params_initial_condition", &
                           "initial_perturbation", &
                           initial_perturbation_local)
            call close_nml(iunit, "params_initial_condition")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_initial_condition(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
