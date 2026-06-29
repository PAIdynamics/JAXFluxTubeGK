submodule (test_params_m) test_params_bsg_s
    ! Submodule that contains helpers for unit tests to initialize BSG
    ! with non-default parameters.
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_bsg(comm, rank, &
                                     vplen_markers, radial_markers, &
                                     num_blocks, bsg_interp_order)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        real(kind=GP), dimension(MAX_BLOCKS), intent(in) :: vplen_markers
        real(kind=GP), dimension(MAX_BLOCKS), intent(in) :: radial_markers
        integer, optional, intent(in) :: num_blocks
        integer, optional, intent(in) :: bsg_interp_order

        integer :: num_blocks_local
        integer :: bsg_interp_order_local

        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        num_blocks_local       = get_num_bsg_blocks()
        bsg_interp_order_local = get_bsg_interp_order()

        if(present(num_blocks)) num_blocks_local = num_blocks
        if(present(bsg_interp_order)) bsg_interp_order_local = bsg_interp_order

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()

            call open_nml(iunit, "params_bsg")
            call write_nml(iunit, "params_bsg", "num_blocks", num_blocks_local)
            call write_nml(iunit, "params_bsg", "vplen_markers", &
                           vplen_markers(1:num_blocks_local))
            call write_nml(iunit, "params_bsg", "radial_markers", &
                           radial_markers(1:num_blocks_local-1))
            call write_nml(iunit, "params_bsg", "bsg_interp_order", &
                           bsg_interp_order_local)
            call close_nml(iunit, "params_bsg")

            close(iunit)

        end if
        call mpi_barrier(comm, ierr)

        ! Read BSG parameters
        call read_params_bsg(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine setup_test_bsg

end submodule
