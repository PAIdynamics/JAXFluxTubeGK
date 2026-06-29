submodule (diag_files_m) em_fields_file_s
    use mpi
    use genex_fortran_env_m, only: MPI_GP
    use params_gyrokinetic_system_m, only: get_with_bpar
    use params_mesh_m, only: get_use_vspectral
    implicit none

contains

    module subroutine initialize_em_fields(this, dcomm_handler, filename, &
                                           n_points_RZ, n_points_phi)
        class(em_fields_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_points_RZ, n_points_phi
        integer :: i, j, k, ierr, dataset_dims(3)

        this%dcomm_handler => dcomm_handler
        this%filename = filename
        this%n_points_RZ = n_points_RZ
        this%n_points_phi = n_points_phi

        ! Setup buffers for the MPI communication
        allocate(this%send_buffer(n_points_RZ, n_points_phi, 3))
        allocate(this%recv_buffer(n_points_RZ, n_points_phi, 3))

        ! First touch initialize recv_buffer, send_buffer is initialized
        ! in the write subroutine.
        !$omp parallel firstprivate(n_points_phi, n_points_RZ) &
        !$omp shared(this) private(i, j, k) default(none)
        do j = 1, 3
           do k = 1, n_points_phi
                !$omp do schedule(static)
                do i = 1, n_points_RZ
                    this%recv_buffer(i, k, j) = 0.0_GP
                enddo
                !$omp end do nowait
            enddo
        enddo
        !$omp end parallel

        if(dcomm_handler%is_master()) then
            ! Create NetCDF file
            ierr = nf90_create(this%filename, NF90_NETCDF4, &
                               this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define time dimension
            ierr = nf90_def_dim(this%file_id, 'dim_time',  NF90_UNLIMITED, &
                                this%dim_time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define RZ dimension
            ierr = nf90_def_dim(this%file_id, 'dim_RZ',  n_points_RZ, &
                                this%dim_RZ_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! Define phi dimension
            ierr = nf90_def_dim(this%file_id, 'dim_phi',  n_points_phi, &
                                this%dim_phi_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define time
            ierr = nf90_def_var(this%file_id, 'time',  NF90_GP, &
                                this%dim_time_id, this%time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Define the dimension for the dataset
            dataset_dims(1) = this%dim_RZ_id
            dataset_dims(2) = this%dim_phi_id
            dataset_dims(3) = this%dim_time_id

            ! Electrostatic potential
            ierr = nf90_def_var(this%file_id, 'es_pot', NF90_GP, &
                                dataset_dims, this%es_pot_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! A_par
            ierr = nf90_def_var(this%file_id, 'A_par', NF90_GP, &
                                dataset_dims, this%A_par_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! B_par
            ! NOTE: B_par is not saved in vspec for now
            if(get_with_bpar() .and. .not. get_use_vspectral()) then
                ierr = nf90_def_var(this%file_id, 'B_par', NF90_GP, &
                                    dataset_dims, this%B_par_id)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            end if

            ierr = nf90_enddef(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            this%write_counter = 1
        endif
    end subroutine

    module subroutine write_em_fields(this, time, lb_stripped, ub_stripped, &
                                      es_pot, A_par, B_par)
        class(em_fields_file_t), intent(inout) :: this
        real(kind = GP), intent(in) :: time
        integer, dimension(2), intent(in) :: lb_stripped
        integer, dimension(2), intent(in) :: ub_stripped
        real(kind = GP), intent(in) :: es_pot( &
            lb_stripped(1) :, &
            lb_stripped(2) : )
        real(kind = GP), intent(in) :: A_par( &
            lb_stripped(1) :, &
            lb_stripped(2) : )
        real(kind = GP), intent(in) :: B_par( &
            lb_stripped(1) :, &
            lb_stripped(2) : )
        integer :: i, j, k, ierr, start(3), start_time(1)

        ! Collect the data on the master process
        !$omp parallel shared(this, es_pot, A_par, B_par) private(i, j, k) &
        !$omp firstprivate(lb_stripped, ub_stripped) default(none)
        do k = 1, this%n_points_phi
            !$omp do schedule(static)
            do i = 1, this%n_points_RZ
                if(      i >= lb_stripped(1) .and. i <= ub_stripped(1) &
                   .and. k >= lb_stripped(2) .and. k <= ub_stripped(2)) then
                    this%send_buffer(i, k, 1) = es_pot(i, k)
                    this%send_buffer(i, k, 2) = A_par(i, k)
                    this%send_buffer(i, k, 3) = B_par(i, k)
                else
                    this%send_buffer(i, k, 1) = 0.0_GP
                    this%send_buffer(i, k, 2) = 0.0_GP
                    this%send_buffer(i, k, 3) = 0.0_GP
                endif
            enddo
            !$omp end do nowait
        enddo
        !$omp end parallel

        call MPI_Reduce(this%send_buffer, this%recv_buffer, &
                        size(this%send_buffer), MPI_GP, MPI_SUM, &
                        0, this%dcomm_handler%get_comm_phi(), ierr)

        if(this%dcomm_handler%is_master()) then
            start_time(1) = this%write_counter
            ierr = nf90_put_var(this%file_id, this%time_id, time, &
                                start = start_time)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            start(1) = 1
            start(2) = 1
            start(3) = this%write_counter

            ! Electrostatic potential
            ierr = nf90_put_var(this%file_id, this%es_pot_id, &
                                this%recv_buffer(:, :, 1), &
                                start=start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Induced parallel electric field
            ierr = nf90_put_var(this%file_id, this%A_par_id, &
                                this%recv_buffer(:, :, 2), &
                                start=start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Write parallel magnetic fluctuations only if enabled and supported
            if(get_with_bpar() .and. .not. get_use_vspectral()) then
                ierr = nf90_put_var(this%file_id, this%B_par_id, &
                                    this%recv_buffer(:, :, 3), &
                                    start=start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            end if

            ierr = nf90_sync(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif

        this%write_counter = this%write_counter + 1
    end subroutine

    module subroutine finalize_em_fields(this)
        type(em_fields_file_t), intent(inout) :: this
        integer :: ierr

        if(this%dcomm_handler%is_master()) then
            ierr = nf90_close(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
