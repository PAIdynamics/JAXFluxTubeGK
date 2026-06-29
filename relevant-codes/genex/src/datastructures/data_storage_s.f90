submodule (data_storage_m) data_storage_s
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_WRN_DATA_STRUCTURE
    use dimensions_m, only: DIM_RZ, DIM_PHI, DIM_VP, DIM_MU, DIM_SP

    implicit none

contains

    integer pure function get_number_of_mail_partners(n_ghost_cells, &
                                                      n_data_cells)
        !! Determines the number of mail partners from the number of ghost
        !! and data cells. We only support 2 or 4 mail partners.
        integer, intent(in) :: n_ghost_cells
        integer, intent(in) :: n_data_cells

        if(n_ghost_cells <= n_data_cells) then
            get_number_of_mail_partners = 2
        else
            get_number_of_mail_partners = 4
        endif
    end function

    module subroutine initialize_parent(this, dcomm_handler, dimensions)
        class(data_storage_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        integer, intent(in) :: dimensions

        integer :: ind

        this%dimensions = dimensions
        this%dcomm_handler => dcomm_handler

        allocate(this%lb(dimensions))
        allocate(this%ub(dimensions))
        allocate(this%lb_stripped(dimensions))
        allocate(this%ub_stripped(dimensions))
        allocate(this%number_of_elements(dimensions))
        allocate(this%number_of_ghost_cells(dimensions))
        allocate(this%number_of_data_cells(dimensions))
        allocate(this%number_of_mail_partners(dimensions))
        allocate(this%dim_permut(dimensions))

        this%size = 1
        do ind = 1, dimensions
            this%lb(ind) = this%dcomm_handler%get_lbound(ind)
            this%ub(ind) = this%dcomm_handler%get_ubound(ind)
            this%lb_stripped(ind) = this%dcomm_handler%get_lbound_stripped(ind)
            this%ub_stripped(ind) = this%dcomm_handler%get_ubound_stripped(ind)
            this%number_of_elements(ind) = &
                this%dcomm_handler%get_number_of_elements(ind)
            this%number_of_ghost_cells(ind) = &
                this%dcomm_handler%get_number_of_ghosts(ind)
            this%dim_permut(ind) = this%dcomm_handler%get_dim_permut(ind)
            this%number_of_data_cells(ind) = &
                this%dcomm_handler%get_number_of_data_elements(ind)
            this%size = this%size * this%number_of_elements(ind)
        enddo

        ! Check the number of MPI ghosts in each dimension
        ! NOTE: We currently support a maximum of 2 ghost cells.
        !       This is assumed in calculating the number of mail partners and
        !       in the packing/unpacking routines of all data_storage types.
        do ind = 1, dimensions
            ! We only give a warning since some tests use more ghost cells
            ! to test different aspects of the data storage.
            if(this%number_of_ghost_cells(ind) > 2) then
                call handle_error("Number of ghost cells was > 2!", &
                                  GENEX_WRN_DATA_STRUCTURE, &
                                  __LINE__, __FILE__, &
                                  additional_info=error_info_t(&
                                    "Index, nghost:", &
                                    [ind, this%number_of_ghost_cells(ind)]))
            endif
        enddo

        ! Calculate the number of mail partners for the mail delivery
        ! NOTE: No MPI communication in RZ and SP needed
        ! TODO: Maybe shift this to children
        if(dimensions == 2 .or. dimensions == 4) then
            this%number_of_mail_partners(this%dim_permut(1)) = 0
            this%number_of_mail_partners(this%dim_permut(2)) = &
                get_number_of_mail_partners( &
                    this%number_of_ghost_cells(this%dim_permut(2)), &
                    this%number_of_data_cells(this%dim_permut(2)))
        else if(dimensions == 5) then
            this%number_of_mail_partners(this%dim_permut(1)) = 0
            this%number_of_mail_partners(this%dim_permut(5)) = 0
            do ind = 2, 4
                this%number_of_mail_partners(this%dim_permut(ind)) = &
                    get_number_of_mail_partners( &
                        this%number_of_ghost_cells(this%dim_permut(ind)), &
                        this%number_of_data_cells(this%dim_permut(ind)))
            enddo
        endif

        this%is_initialized = .true.
    end subroutine

end submodule
