submodule (test_params_m) test_params_gyrokinetic_system_s
    ! Submodule that contains helpers for unit tests to initialize
    ! the gyrokinetic system with non-default parameters.
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_gyrokinetic_system(comm, rank, &
                                                    with_nlin_polarization, &
                                                    with_bpar)
        !! Initializes gyrokinetic system with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        logical, optional, intent(in) :: with_nlin_polarization
        logical, optional, intent(in) :: with_bpar

        logical, allocatable :: with_nlin_polarization_local
        logical, allocatable :: with_bpar_local

        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        with_nlin_polarization_local = get_with_nlin_polarization()
        with_bpar_local = get_with_bpar()

        if(present(with_nlin_polarization)) &
            with_nlin_polarization_local = with_nlin_polarization
        if(present(with_bpar)) &
            with_bpar_local = with_bpar

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", &
                 iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_gyrokinetic_system")
            call write_nml(iunit, "params_gyrokinetic_system", &
                           "with_nlin_polarization", &
                           with_nlin_polarization_local)
            call write_nml(iunit, "params_gyrokinetic_system", &
                           "with_bpar", &
                           with_bpar_local)
            call close_nml(iunit, "params_gyrokinetic_system")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_gyrokinetic_system(&
                trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
