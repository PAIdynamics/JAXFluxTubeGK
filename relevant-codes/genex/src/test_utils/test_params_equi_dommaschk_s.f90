submodule (test_params_m) test_params_equi_dommaschk_s
    ! Submodule that contains helpers for unit tests to initialize the dommaschk
    ! equilibrium with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_equi_dommaschk(comm, rank, &
                                                m_tor_consecutive, &
                                                l_pol, &
                                                num_field_periods, &
                                                fitting_coef)
        !! Initializes the simulation with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, optional, intent(in) :: m_tor_consecutive
        integer, optional, intent(in) :: l_pol
        integer, optional, intent(in) :: num_field_periods
        real(kind=GP), dimension(:,:,:), optional, intent(in) :: fitting_coef

        integer :: m_tor_consecutive_local, l_pol_local, &
                   num_field_periods_local
        real(kind=GP), dimension(:,:,:), allocatable :: fitting_coef_local

        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        m_tor_consecutive_local       = get_dommaschk_m_tor_consecutive()
        l_pol_local                   = get_dommaschk_l_pol()
        num_field_periods_local       = get_dommaschk_num_field_periods()
        fitting_coef_local            = get_dommaschk_fitting_coef()

        if(present(m_tor_consecutive)) &
            m_tor_consecutive_local = m_tor_consecutive
        if(present(l_pol)) l_pol_local = l_pol
        if(present(num_field_periods)) &
            num_field_periods_local = num_field_periods
        if(present(fitting_coef)) fitting_coef_local = fitting_coef

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_equi_dommaschk")

            call write_nml(iunit, "params_equi_dommaschk", &
                           "m_tor_consecutive", m_tor_consecutive_local)
            call write_nml(iunit, "params_equi_dommaschk", "l_pol", &
                           l_pol_local)
            call write_nml(iunit, "params_equi_dommaschk", &
                           "num_field_periods", num_field_periods_local)

            call close_nml(iunit, "params_equi_dommaschk")

            call open_nml(iunit, "params_equi_dommaschk_fitting_coef")

            call write_nml(iunit, "params_equi_dommaschk_fitting_coef", &
                           "fitting_coef", fitting_coef_local)

            call close_nml(iunit, "params_equi_dommaschk_fitting_coef")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_dommaschk(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
