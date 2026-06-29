submodule (test_params_m) test_params_mesh_s
    ! Submodule that contains helpers for unit tests to initialize mesh
    ! with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_mesh(comm, rank, equilibrium_type, &
                                      spacing_RZ, n_points_phi, &
                                      n_points_vp, n_points_mu, &
                                      n_points_sp, length_vp, length_mu, &
                                      quad_type_vp, grid_type_mu, &
                                      n_levels, use_vspectral, &
                                      only_first_field_period, &
                                      use_bsg)
        !! Initialize the simulation with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, optional, intent(in) :: equilibrium_type
        real(kind=GP), optional, intent(in) :: spacing_RZ
        integer, optional, intent(in) :: n_points_phi
        integer, optional, intent(in) :: n_points_vp
        integer, optional, intent(in) :: n_points_mu
        integer, optional, intent(in) :: n_points_sp
        real(kind=GP), optional, intent(in) :: length_vp
        real(kind=GP), optional, intent(in) :: length_mu
        character(len=*), optional, intent(in) :: quad_type_vp
        character(len=*), optional, intent(in) :: grid_type_mu
        integer, optional, intent(in) :: n_levels
        logical, optional, intent(in) :: use_vspectral
        logical, optional, intent(in) :: only_first_field_period
        logical, optional, intent(in) :: use_bsg

        integer :: equilibrium_type_local, n_points_phi_local, &
                   n_points_vp_local, n_points_mu_local, n_points_sp_local, &
                   n_levels_local
        real(kind=GP) :: spacing_RZ_local, length_vp_local, length_mu_local
        character(len=:), allocatable :: quad_type_vp_local, grid_type_mu_local
        logical :: use_vspectral_local, only_first_field_period_local
        logical :: use_bsg_local

        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        equilibrium_type_local        = get_equilibrium_type()
        spacing_RZ_local              = get_spacing_RZ()
        n_points_phi_local            = get_n_points_phi()
        n_points_vp_local             = get_n_points_vp()
        n_points_mu_local             = get_n_points_mu()
        n_points_sp_local             = get_n_points_sp()
        length_vp_local               = get_length_vp()
        length_mu_local               = get_length_mu()
        quad_type_vp_local            = get_quad_type_vp()
        grid_type_mu_local            = get_grid_type_mu()
        n_levels_local                = get_n_levels()
        use_vspectral_local           = get_use_vspectral()
        use_bsg_local                 = get_use_bsg()
        only_first_field_period_local = get_only_first_field_period()

        if(present(equilibrium_type)) equilibrium_type_local = equilibrium_type
        if(present(spacing_RZ)) spacing_RZ_local = spacing_RZ
        if(present(n_points_phi)) n_points_phi_local = n_points_phi
        if(present(n_points_vp)) n_points_vp_local = n_points_vp
        if(present(n_points_mu)) n_points_mu_local = n_points_mu
        if(present(n_points_sp)) n_points_sp_local = n_points_sp
        if(present(length_vp)) length_vp_local = length_vp
        if(present(length_mu)) length_mu_local = length_mu
        if(present(quad_type_vp)) quad_type_vp_local = quad_type_vp
        if(present(grid_type_mu)) grid_type_mu_local = grid_type_mu
        if(present(n_levels)) n_levels_local = n_levels
        if(present(use_vspectral)) use_vspectral_local = use_vspectral
        if(present(only_first_field_period)) &
            only_first_field_period_local = only_first_field_period
        if(present(use_bsg)) use_bsg_local = use_bsg

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_mesh")

            call write_nml(iunit, "params_mesh", "equilibrium_type", &
                           equilibrium_type_local)
            call write_nml(iunit, "params_mesh", "spacing_RZ", &
                           spacing_RZ_local)
            call write_nml(iunit, "params_mesh", "n_points_phi", &
                           n_points_phi_local)
            call write_nml(iunit, "params_mesh", "n_points_vp", &
                           n_points_vp_local)
            call write_nml(iunit, "params_mesh", "n_points_mu", &
                           n_points_mu_local)
            call write_nml(iunit, "params_mesh", "n_points_sp", &
                           n_points_sp_local)
            call write_nml(iunit, "params_mesh", "length_vp", &
                           length_vp_local)
            call write_nml(iunit, "params_mesh", "length_mu", &
                           length_mu_local)
            call write_nml(iunit, "params_mesh", "quad_type_vp", &
                           quad_type_vp_local)
            call write_nml(iunit, "params_mesh", "grid_type_mu", &
                           grid_type_mu_local)
            call write_nml(iunit, "params_mesh", "n_levels", &
                           n_levels_local)
            call write_nml(iunit, "params_mesh", "use_vspectral", &
                           use_vspectral_local)
            call write_nml(iunit, "params_mesh", "only_first_field_period", &
                           only_first_field_period_local)
            call write_nml(iunit, "params_mesh", "use_bsg", use_bsg_local)

            call close_nml(iunit, "params_mesh")

            close(iunit)

        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_mesh(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine setup_test_mesh

end submodule
