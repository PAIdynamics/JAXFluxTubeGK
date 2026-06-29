submodule (test_params_m) test_params_equi_slab_s
    ! Submodule that contains helpers for unit tests to initialize the slab
    ! equilibrium with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_equi_slab(comm, rank, boxsize, sol, yperiodic)
        !! Initializes the simulation with the given test parameters
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        real(kind=GP), optional, intent(in) :: boxsize
        logical, optional, intent(in) :: sol
        logical, optional, intent(in) :: yperiodic

        real(kind=GP) :: boxsize_local
        logical :: sol_local, yperiodic_local
        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        boxsize_local   = get_slab_boxsize()
        sol_local       = get_slab_sol()
        yperiodic_local = get_slab_yperiodic()

        if(present(boxsize)) boxsize_local = boxsize
        if(present(sol)) sol_local = sol
        if(present(yperiodic)) yperiodic_local = yperiodic

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "equi_slab_params")

            call write_nml(iunit, "equi_slab_params", "boxsize", &
                           boxsize_local)
            call write_nml(iunit, "equi_slab_params", "sol", &
                           sol_local)
            call write_nml(iunit, "equi_slab_params", "yperiodic", &
                           yperiodic_local)

            call close_nml(iunit, "equi_slab_params")

            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_slab(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
