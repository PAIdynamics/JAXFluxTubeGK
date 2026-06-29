submodule(test_params_m) test_params_neutrals_init_s
    ! Submodule that contains helpers for unit tests to initialize the
    ! neutrals initial conditions with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_neutrals_init(comm, rank, o, &
                                               params)
        !! Initialize the simulation with the given test parameters for the
        !! neutrals initial conditions parameters
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        integer, intent(in) :: o
        type(params_neutrals_init_t), intent(in) :: params

        type(params_neutrals_init_t) :: params_local
        integer :: iunit, io_error, ierr
        character(:), allocatable :: neutrals_file

        params_local = params

        neutrals_file = trim(get_test_params_file()) &
                      //"_"//trim(get_neut_name(o))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=neutrals_file, &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_neutrals_init")
            call write_nml(iunit, "params_neutrals_init", &
                           "profile_type_dens", &
                            params_local%profile_type_dens)
            call write_nml(iunit, "params_neutrals_init", &
                           "initial_perturbation_dens", &
                           params_local%initial_perturbation_dens)
            call close_nml(iunit, "params_neutrals_init")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read neutrals init parameters
        call read_params_neutrals_init(neutrals_file, o)
        call mpi_barrier(comm, ierr)
    end subroutine
end submodule
