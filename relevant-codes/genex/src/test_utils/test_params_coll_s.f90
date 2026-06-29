submodule (test_params_m) test_params_coll_s
    ! Submodule that contains helpers for unit tests to initialize collisions
    ! with non-default parameters.
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_coll(comm, rank, coll_type, relx_type)
        !! Initializes the simulation with the given test parameters for the
        !! collision operator
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        character(len=*), optional, intent(in) :: coll_type
        character(len=*), optional, intent(in) :: relx_type

        character(len=:), allocatable :: coll_type_local
        character(len=:), allocatable :: relx_type_local

        integer :: iunit, io_error, ierr

        coll_type_local = get_coll_type()
        relx_type_local = get_relx_type()

        if(present(coll_type)) coll_type_local = coll_type
        if(present(relx_type)) relx_type_local = relx_type

        ! Let master proc write the file
        if(rank == 0) then

           open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                action='write', status="replace", &
                iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_collisions")
            call write_nml(iunit, "params_collisions", "coll_type", &
                           coll_type_local)
            call write_nml(iunit, "params_collisions", "relx_type", &
                           relx_type_local)
            call close_nml(iunit, "params_collisions")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_collisions(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
