submodule(test_params_m) test_params_sources_s
    ! Submodule that contains helpers for unit tests to initialize the
    ! sources with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_source_dens(comm, rank, n, params)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        integer, intent(in) :: n
        type(params_source_loc_t), intent(in) :: params

        type(params_source_loc_t) :: params_local
        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_source_dens")
            call write_nml(iunit, "params_source_dens", &
                           "temp", &
                           params_local%temp)
            call write_nml(iunit, "params_source_dens", &
                           "rho_mid", &
                           params_local%rho_mid)
            call write_nml(iunit, "params_source_dens", &
                           "width", &
                            params_local%width)
            call write_nml(iunit, "params_source_dens", &
                           "amp", &
                           params_local%amp)
            call write_nml(iunit, "params_source_dens", &
                           "is_pure", &
                           params_local%is_pure)
            call close_nml(iunit, "params_source_dens")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        call read_params_source_dens(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

    module subroutine setup_test_source_torque(comm, rank, n, params)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        integer, intent(in) :: n
        type(params_source_loc_t), intent(in) :: params

        type(params_source_loc_t) :: params_local
        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_source_torque")
            call write_nml(iunit, "params_source_torque", &
                           "temp", &
                           params_local%temp)
            call write_nml(iunit, "params_source_torque", &
                           "rho_mid", &
                           params_local%rho_mid)
            call write_nml(iunit, "params_source_torque", &
                           "width", &
                            params_local%width)
            call write_nml(iunit, "params_source_torque", &
                           "amp", &
                           params_local%amp)
            call write_nml(iunit, "params_source_torque", &
                           "is_pure", &
                           params_local%is_pure)
            call close_nml(iunit, "params_source_torque")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        call read_params_source_torque(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

    module subroutine setup_test_source_heat(comm, rank, n, params)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        integer, intent(in) :: n
        type(params_source_loc_t), intent(in) :: params

        type(params_source_loc_t) :: params_local
        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_source_heat")
            call write_nml(iunit, "params_source_heat", &
                           "temp", &
                           params_local%temp)
            call write_nml(iunit, "params_source_heat", &
                           "rho_mid", &
                           params_local%rho_mid)
            call write_nml(iunit, "params_source_heat", &
                           "width", &
                            params_local%width)
            call write_nml(iunit, "params_source_heat", &
                           "amp", &
                           params_local%amp)
            call write_nml(iunit, "params_source_heat", &
                           "is_pure", &
                           params_local%is_pure)
            call close_nml(iunit, "params_source_heat")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        call read_params_source_heat(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
