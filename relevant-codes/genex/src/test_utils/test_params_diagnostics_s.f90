submodule(test_params_m) test_params_diagnostics_s
    ! Submodule that contains helpers for unit tests to initialize the
    ! diagnostics with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_diagnostics(comm, rank, &
                                             polar_n_theta, polar_n_rho, &
                                             diagnose_tpc)
        !! Initialize the simulation with the given test parameters for the
        !! diagnostics parameters
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        integer, optional, intent(in) :: polar_n_theta
        integer, optional, intent(in) :: polar_n_rho
        logical, optional, intent(in) :: diagnose_tpc

        integer :: polar_n_theta_local, polar_n_rho_local
        logical :: diagnose_tpc_local
        integer :: n, iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        polar_n_theta_local = get_polar_n_theta()
        polar_n_rho_local   = get_polar_n_rho()
        diagnose_tpc_local   = get_diagnose_tpc()

        if(present(polar_n_theta)) polar_n_theta_local = polar_n_theta
        if(present(polar_n_rho))   polar_n_rho_local   = polar_n_rho
        if(present(diagnose_tpc))   diagnose_tpc_local   = diagnose_tpc

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_diagnostics")

            call write_nml(iunit, "params_diagnostics", "polar_n_theta", &
                           polar_n_theta_local)
            call write_nml(iunit, "params_diagnostics", "polar_n_rho", &
                           polar_n_rho_local)
            call write_nml(iunit, "params_diagnostics", "diagnose_tpc", &
                           diagnose_tpc_local)

            call close_nml(iunit, "params_diagnostics")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_diagnostics(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)
    end subroutine
end submodule
