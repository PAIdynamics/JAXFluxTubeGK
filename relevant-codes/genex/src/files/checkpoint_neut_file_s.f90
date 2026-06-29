submodule (diag_files_m) checkpoint_neut_file_s
    use mpi
    use type_converters_m, only: string
    use params_neutrals_m, only: get_n_points_neut
    implicit none

contains

    function remove_filename_ending(input_file) result(output_file)
        character(len=*), intent(in) :: input_file
        character(len=:), allocatable :: output_file
        integer :: idx

        idx = scan(trim(input_file), ".", BACK=.true.)
        if (idx > 1) then
            output_file = input_file(1:idx-1)
        else
            output_file = input_file
        endif
    end function

    integer module function calculate_process_id_neut(this)
        class(checkpoint_neut_file_t), intent(in) :: this
        integer :: i, cum_prod(5), coords(5)

        cum_prod(1) = this%dcomm_handler%get_n_procs_RZ()
        cum_prod(2) = cum_prod(1) * this%dcomm_handler%get_n_procs_phi()
        cum_prod(3) = cum_prod(2) * this%dcomm_handler%get_n_procs_vp()
        cum_prod(4) = cum_prod(3) * this%dcomm_handler%get_n_procs_mu()
        cum_prod(5) = cum_prod(4) * this%dcomm_handler%get_n_procs_sp()

        coords = this%dcomm_handler%get_cart_coords()

        calculate_process_id_neut = coords(1) + 1
        do i = 2, 5
            calculate_process_id_neut = calculate_process_id_neut &
                                      + coords(i) * cum_prod(i - 1)
        enddo
    end function

    module subroutine initialize_checkpoint_neut(this, dcomm_handler, &
            filename, n_points_RZ, n_points_phi, n_points_mom, &
            n_points_sp, from_existing)
        class(checkpoint_neut_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_points_RZ, n_points_phi, n_points_mom, &
                               n_points_sp
        logical, optional, intent(in) :: from_existing

        integer :: ierr, dataset_dims(5)
        logical :: from_existing_loc

        if(present(from_existing)) then
            from_existing_loc = from_existing
        else
            from_existing_loc = .false.
        endif

        this%dcomm_handler => dcomm_handler

        this%filename = remove_filename_ending(filename) // "_" &
                        // string(this%calculate_process_id_neut()) // ".nc"

        if (from_existing_loc) then
            ! Open existing checkpoint_neut_file for read and write access
            ierr = nf90_open(this%filename, NF90_WRITE, this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! TODO: Check if dimensions are correct

            ierr = nf90_inq_dimid(this%file_id, 'dim_time', this%dim_time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_inq_varid(this%file_id, 'lb_stripped', &
                                  this%lb_stripped_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_inq_varid(this%file_id, 'ub_stripped', &
                                  this%ub_stripped_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_inq_varid(this%file_id, 'time', this%time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_inq_varid(this%file_id, 'neutrals_state', &
                                  this%neutrals_state_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Get the number of records in the file
            ierr = nf90_inquire_dimension(this%file_id, this%dim_time_id, &
                                          len=this%write_counter)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            this%write_counter = this%write_counter + 1
        else
            ! Create new checkpoint file
            ierr = nf90_create(this%filename, NF90_NETCDF4, this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define time dimension
            ierr = nf90_def_dim(this%file_id, 'dim_time',  NF90_UNLIMITED, &
                                this%dim_time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define RZ dimension
            ierr = nf90_def_dim(this%file_id, 'dim_RZ', &
                                n_points_RZ, &
                                this%dim_RZ_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! Define phi dimension
            ierr = nf90_def_dim(this%file_id, 'dim_phi', &
                                n_points_phi / dcomm_handler%get_n_procs_phi(),&
                                this%dim_phi_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! Define mom dimension
            ierr = nf90_def_dim(this%file_id, 'dim_mom', &
                                1, &
                                this%dim_mom_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! Define sp dimension
            ierr = nf90_def_dim(this%file_id, 'dim_sp', &
                                get_n_points_neut(), &
                                this%dim_sp_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define boundaries dimension
            ierr = nf90_def_dim(this%file_id, 'dim_bound', 4, &
                                this%dim_bound_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define lb_stripped
            ierr = nf90_def_var(this%file_id, 'lb_stripped', NF90_INT, &
                                this%dim_bound_id, this%lb_stripped_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define ub_stripped
            ierr = nf90_def_var(this%file_id, 'ub_stripped', NF90_INT, &
                                this%dim_bound_id, this%ub_stripped_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define time
            ierr = nf90_def_var(this%file_id, 'time',  NF90_GP, &
                                this%dim_time_id, this%time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define the dimension for the dataset
            dataset_dims(1) = this%dim_RZ_id
            dataset_dims(2) = this%dim_phi_id
            dataset_dims(3) = this%dim_mom_id
            dataset_dims(4) = this%dim_sp_id
            dataset_dims(5) = this%dim_time_id

            ierr = nf90_def_var(this%file_id, 'neutrals_state', NF90_GP, &
                                dataset_dims, this%neutrals_state_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_enddef(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            this%write_counter = 1
        end if
    end subroutine

    module subroutine write_checkpoint_neut(this, time, neutrals_state)
        class(checkpoint_neut_file_t), intent(inout) :: this
        real(kind=GP), intent(in) :: time
        real(kind=GP), pointer, dimension(:,:,:,:), intent(in) :: neutrals_state

        integer :: ierr
        ! NetCDF error status
        integer :: start(5), start_time(1)
        ! NetCDF index arrays where the first of the data values be written
        integer, dimension(4) :: lb_stripped, ub_stripped
        ! The lower and upper boundaries of the neutrals state

        ! Get the lower and upper boundaries from neutrals_state
        lb_stripped = lbound(neutrals_state)
        ub_stripped = ubound(neutrals_state)

        if (this%write_counter == 1) then
            ! Write lower boundaries to checkpoint files at initial
            ierr = nf90_put_var(this%file_id, this%lb_stripped_id, &
                                lb_stripped, start=[1, 1])
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Write upper boundaries to checkpoint files at initial
            ierr = nf90_put_var(this%file_id, this%ub_stripped_id, &
                                ub_stripped, start=[1, 1])
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        end if

        ! Write time
        start_time(1) = this%write_counter
        ierr = nf90_put_var(this%file_id, this%time_id, time, &
                            start=start_time)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        start = [1, 1, 1, 1, start_time(1)]

        ! Write neutrals state
        ierr = nf90_put_var(this%file_id, this%neutrals_state_id, &
                            neutrals_state, &
                            start=start)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_sync(this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        this%write_counter = this%write_counter + 1
    end subroutine

    module subroutine read_checkpoint_neut(this, time, neutrals_state)
        class(checkpoint_neut_file_t), intent(inout) :: this
        real(kind=GP), intent(out) :: time
        real(kind=GP), pointer, dimension(:,:,:,:), &
                                                intent(inout) :: neutrals_state

        integer :: ierr
        ! NetCDF error status
        integer :: start(5), start_time(1)
        ! NetCDF index arrays where the first of the data values be read
        integer, dimension(4) :: lb_stripped_file, ub_stripped_file
        ! The lower and upper boundaries read from checkpoint files
        integer, dimension(4) :: lb_stripped_ref, ub_stripped_ref
        ! The lower and upper boundaries obtained from neutrals_state
        ! as reference

        ! Read lower boundaries from checkpoint files
        ierr = nf90_get_var(this%file_id, this%lb_stripped_id, &
                            lb_stripped_file, start=[1, 1])
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Read upper boundaries from checkpoint files
        ierr = nf90_get_var(this%file_id, this%ub_stripped_id, &
                            ub_stripped_file, start=[1, 1])
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Read time
        start_time(1) = this%write_counter - 1
        ierr = nf90_get_var(this%file_id, this%time_id, time, &
                            start=start_time)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        start = [1, 1, 1, 1, start_time(1)]

        ! Read the neutrals state
        ierr = nf90_get_var(this%file_id, this%neutrals_state_id, &
                            neutrals_state, &
                            start=start)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get the lower and upper boundaries from neutrals_state for reference
        lb_stripped_ref = lbound(neutrals_state)
        ub_stripped_ref = ubound(neutrals_state)

        ! Check whether the lower and upper boundaries read from the file
        ! match the input boundaries.
        call this%check_array(lb_stripped_file, lb_stripped_ref, __LINE__, &
                              __FILE__)
        call this%check_array(ub_stripped_file, ub_stripped_ref, __LINE__, &
                              __FILE__)
    end subroutine

    module subroutine finalize_checkpoint_neut(this)
        type(checkpoint_neut_file_t), intent(inout) :: this
        integer :: ierr

        ierr = nf90_close(this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

end submodule
