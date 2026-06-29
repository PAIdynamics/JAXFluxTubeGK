submodule (test_params_m) test_params_bnd_cond_s
    ! Submodule that contains helpers for unit test to
    ! initialize the boundary condition parameters.
    use logger_m, only: logger_get_debug_channel
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_bnd_cond(comm, rank, &
                                          bnd_cond_type, &
                                          freeze_even_mom, &
                                          freeze_dens)
        !! Initialize the simulation with the given test parameters for the
        !! bondary conditions
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        character(len=*), optional, intent(in) :: bnd_cond_type
        !! Type of boundary condition
        logical, optional, intent(in) :: freeze_even_mom
        !! Flag to freeze the boundary condition of even moments
        logical, optional, intent(in) :: freeze_dens
        !! Flag to freeze the boundary condition of density

        character(:), allocatable :: bnd_cond_type_local
        logical :: freeze_even_mom_local, freeze_dens_local
        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        freeze_even_mom_local = get_freeze_even_mom()
        freeze_dens_local = get_freeze_dens()
        bnd_cond_type_local = get_bnd_cond_type()

        if(present(freeze_even_mom)) &
            freeze_even_mom_local = freeze_even_mom
        if(present(freeze_dens)) &
            freeze_dens_local = freeze_dens
        if(present(bnd_cond_type)) &
            bnd_cond_type_local = bnd_cond_type

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_bnd_cond")
            call write_nml(iunit, "params_bnd_cond", "bnd_cond_type", &
                           bnd_cond_type_local)
            call write_nml(iunit, "params_bnd_cond", "freeze_even_mom", &
                           freeze_even_mom_local)
            call write_nml(iunit, "params_bnd_cond", "freeze_dens", &
                           freeze_dens_local)
            call close_nml(iunit, "params_bnd_cond")
            close(iunit)
        end if

        call mpi_barrier(comm, ierr)

        ! Read boundary condition parameters
        call read_params_bnd_cond(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine
end submodule
