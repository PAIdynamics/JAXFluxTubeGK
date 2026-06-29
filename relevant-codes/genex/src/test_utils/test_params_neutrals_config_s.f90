submodule(test_params_m) test_params_neutrals_config_s
    ! Submodule that contains helpers for unit tests to initialize the
    ! neutrals configuration with non-default parameters.
    use genex_fortran_env_m, only : GP
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_neutrals_config(comm, rank, &
                                               neutrals_evolve_type, &
                                               neutrals_coupling_type, &
                                               neutrals_dens_floor, &
                                               neutrals_temp_floor, &
                                               neutrals_gamma_u, &
                                               neutrals_gamma_T, &
                                               neutrals_gamma_W)
        !! Initialize the simulation with the given test parameters for the
        !! neutrals configuration parameters
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=*), optional, intent(in) :: neutrals_evolve_type
        character(len=*), optional, intent(in) :: neutrals_coupling_type
        real(kind=GP), optional, intent(in) :: neutrals_dens_floor
        real(kind=GP), optional, intent(in) :: neutrals_temp_floor
        real(kind=GP), optional, intent(in) :: neutrals_gamma_u
        real(kind=GP), optional, intent(in) :: neutrals_gamma_T
        real(kind=GP), optional, intent(in) :: neutrals_gamma_W

        integer :: iunit, io_error, ierr
        character(len=:), allocatable :: neutrals_evolve_type_local
        character(len=:), allocatable :: neutrals_coupling_type_local
        real(kind=GP) :: neutrals_dens_floor_local
        real(kind=GP) :: neutrals_temp_floor_local
        real(kind=GP) :: neutrals_gamma_u_local
        real(kind=GP) :: neutrals_gamma_T_local
        real(kind=GP) :: neutrals_gamma_W_local

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.

        neutrals_evolve_type_local = get_neut_evolve_type()
        neutrals_coupling_type_local = get_neut_coupling_type()
        neutrals_dens_floor_local = get_neut_dens_floor()
        neutrals_temp_floor_local = get_neut_temp_floor()
        neutrals_gamma_u_local = get_neut_gamma_u()
        neutrals_gamma_T_local = get_neut_gamma_T()
        neutrals_gamma_W_local = get_neut_gamma_W()

        if(present(neutrals_evolve_type)) then
            neutrals_evolve_type_local = neutrals_evolve_type
        end if
        if(present(neutrals_coupling_type)) then
            neutrals_coupling_type_local = neutrals_coupling_type
        end if
        if(present(neutrals_dens_floor)) then
            neutrals_dens_floor_local = neutrals_dens_floor
        end if
        if(present(neutrals_temp_floor)) then
            neutrals_temp_floor_local = neutrals_temp_floor
        end if
        if(present(neutrals_gamma_u)) then
            neutrals_gamma_u_local = neutrals_gamma_u
        end if
        if(present(neutrals_gamma_T)) then
            neutrals_gamma_T_local = neutrals_gamma_T
        end if
        if(present(neutrals_gamma_W)) then
            neutrals_gamma_W_local = neutrals_gamma_W
        end if

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_neutrals_config")

            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_evolve_type", &
                            neutrals_evolve_type_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_coupling_type", &
                           neutrals_coupling_type_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_dens_floor", &
                           neutrals_dens_floor_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_temp_floor", &
                           neutrals_temp_floor_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_gamma_u", &
                           neutrals_gamma_u_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_gamma_T", &
                           neutrals_gamma_T_local)
            call write_nml(iunit, "params_neutrals_config", &
                           "neutrals_gamma_W", &
                           neutrals_gamma_W_local)

            call close_nml(iunit, "params_neutrals_config")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_neutrals_config(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)
    end subroutine
end submodule
