submodule (test_params_m) test_params_time_split_s
    use test_params_io_m
    use params_time_split_m, only: get_time_scheme_collisions, &
                                   get_time_scheme_neutrals
    implicit none

contains

    module subroutine setup_test_time_split(comm, rank, &
                                            time_scheme_collisions, &
                                            time_scheme_neutrals)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=*), optional, intent(in) :: time_scheme_collisions
        character(len=*), optional, intent(in) :: time_scheme_neutrals

        character(len=:), allocatable :: time_scheme_collisions_local, &
                                         time_scheme_neutrals_local

        integer :: iunit, io_error, ierr

        time_scheme_collisions_local = get_time_scheme_collisions()
        time_scheme_neutrals_local   = get_time_scheme_neutrals()

        if(present(time_scheme_collisions)) &
            time_scheme_collisions_local = time_scheme_collisions
        if(present(time_scheme_neutrals)) &
            time_scheme_neutrals_local = time_scheme_neutrals

        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_time_split")

            call write_nml(iunit, "params_time_split", &
                           "time_scheme_collisions", &
                           time_scheme_collisions_local)
            call write_nml(iunit, "params_time_split", &
                           "time_scheme_neutrals", &
                           time_scheme_neutrals_local)

            call close_nml(iunit, "params_time_split")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        call read_params_time_split(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
