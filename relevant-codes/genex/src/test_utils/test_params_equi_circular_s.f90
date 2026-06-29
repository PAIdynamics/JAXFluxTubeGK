submodule (test_params_m) test_params_equi_circular_s
    ! Submodule that contains helpers for unit tests to initialize the circular
    ! equilibrium with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_equi_circular(comm, rank, &
                                               rhomin, rhomax, &
                                               qtype, rhoq_ref, q_ref, shear, &
                                               dtheta_limiter, rho_limiter, &
                                               theta_limiter)
        !! Initializes the simulation with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        real(kind=GP), optional, intent(in) :: rhomin
        real(kind=GP), optional, intent(in) :: rhomax
        integer,       optional, intent(in) :: qtype
        real(kind=GP), optional, intent(in) :: rhoq_ref
        real(kind=GP), optional, intent(in) :: q_ref
        real(kind=GP), optional, intent(in) :: shear
        real(kind=GP), optional, intent(in) :: dtheta_limiter
        real(kind=GP), optional, intent(in) :: rho_limiter
        real(kind=GP), optional, intent(in) :: theta_limiter

        real(kind=GP) :: rhomin_local, rhomax_local, &
                         rhoq_ref_local, q_ref_local, shear_local, &
                         dtheta_limiter_local, rho_limiter_local, &
                         theta_limiter_local
        integer :: qtype_local
        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        rhomin_local         = get_circular_rhomin()
        rhomax_local         = get_circular_rhomax()
        qtype_local          = get_circular_qtype()
        rhoq_ref_local       = get_circular_rhoq_ref()
        q_ref_local          = get_circular_q_ref()
        shear_local          = get_circular_shear()
        dtheta_limiter_local = get_circular_dtheta_limiter()
        rho_limiter_local    = get_circular_rho_limiter()
        theta_limiter_local  = get_circular_theta_limiter()

        if(present(rhomin)) rhomin_local = rhomin
        if(present(rhomax)) rhomax_local = rhomax
        if(present(qtype)) qtype_local = qtype
        if(present(rhoq_ref)) rhoq_ref_local = rhoq_ref
        if(present(q_ref)) q_ref_local = q_ref
        if(present(shear)) shear_local = shear
        if(present(dtheta_limiter)) dtheta_limiter_local = dtheta_limiter
        if(present(rho_limiter)) rho_limiter_local = rho_limiter
        if(present(theta_limiter)) theta_limiter_local = theta_limiter

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "equi_circular_params")

            call write_nml(iunit, "equi_circular_params", "rhomin", &
                           rhomin_local)
            call write_nml(iunit, "equi_circular_params", "rhomax", &
                           rhomax_local)
            call write_nml(iunit, "equi_circular_params", "qtype", &
                           qtype_local)
            call write_nml(iunit, "equi_circular_params", "rhoq_ref", &
                           rhoq_ref_local)
            call write_nml(iunit, "equi_circular_params", "q_ref", &
                           q_ref_local)
            call write_nml(iunit, "equi_circular_params", "shear", &
                           shear_local)
            call write_nml(iunit, "equi_circular_params", "dtheta_limiter", &
                           dtheta_limiter_local)
            call write_nml(iunit, "equi_circular_params", "rho_limiter", &
                           rho_limiter_local)
            call write_nml(iunit, "equi_circular_params", "theta_limiter", &
                           theta_limiter_local)

            call close_nml(iunit, "equi_circular_params")

            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_circular(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
