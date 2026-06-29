submodule (diag_files_m) mom_2d_file_s
    use mpi
    use genex_fortran_env_m, only: MPI_GP
    use params_species_m, only: get_name
    implicit none

contains

    module subroutine initialize_mom_2d(this, dcomm_handler, filename, &
                                        n_points_RZ, n_points_phi, n_points_sp)
        class(mom_2d_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_points_RZ, n_points_phi, n_points_sp

        integer :: n, j, k, i, ierr
        integer :: dataset_dims(3)

        this%dcomm_handler => dcomm_handler
        this%filename = filename
        this%n_points_RZ = n_points_RZ
        this%n_points_phi = n_points_phi
        this%n_points_sp = n_points_sp

        allocate(this%species_group(n_points_sp))
        allocate(this%u_par_ids(n_points_sp))
        allocate(this%n_ids(n_points_sp))
        allocate(this%E_par_ids(n_points_sp))
        allocate(this%E_perp_ids(n_points_sp))
        allocate(this%Q_par_ids(n_points_sp))
        allocate(this%Q_perp_ids(n_points_sp))
        allocate(this%K_par_ids(n_points_sp))
        allocate(this%K_perp_ids(n_points_sp))

        ! Setup buffers for the MPI communication
        allocate(this%send_buffer(n_points_RZ, n_points_phi, 8, n_points_sp))
        allocate(this%recv_buffer(n_points_RZ, n_points_phi, 8, n_points_sp))

        ! First touch initialize recv_buffer, send_buffer is initialised
        ! in the write subroutine.
        !$omp parallel firstprivate(n_points_phi, n_points_RZ, n_points_sp) &
        !$omp shared(this) private(k, i, j, n) default(none)
        do n = 1, n_points_sp
            do j = 1, 8
                do k = 1, n_points_phi
                    !$omp do schedule(static)
                    do i = 1, n_points_RZ
                        this%recv_buffer(i, k, j, n) = 0.0_GP
                    enddo
                    !$omp end do nowait
                enddo
            enddo
        enddo
        !$omp end parallel

        if(dcomm_handler%is_master()) then
            ! create parallel NetCDF moments file
            ierr = nf90_create(this%filename, NF90_NETCDF4, this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define time dimension
            ierr = nf90_def_dim(this%file_id, 'dim_time',  NF90_UNLIMITED, &
                                this%dim_time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define RZ dimension
            ierr = nf90_def_dim(this%file_id, 'dim_RZ',  n_points_RZ, &
                                this%dim_RZ_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! define phi dimension
            ierr = nf90_def_dim(this%file_id, 'dim_phi',  n_points_phi, &
                                this%dim_phi_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define time
            ierr = nf90_def_var(this%file_id, 'time',  NF90_GP, &
                                this%dim_time_id, this%time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define the dimension for the dataset
            dataset_dims(1) = this%dim_RZ_id
            dataset_dims(2) = this%dim_phi_id
            dataset_dims(3) = this%dim_time_id
            do n = 1, n_points_sp
                ! create group
                ierr = nf90_def_grp(this%file_id, get_name(n), &
                                    this%species_group(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! particle number
                ierr = nf90_def_var(this%species_group(n), 'n',  &
                                    NF90_GP, dataset_dims, &
                                    this%n_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! u_par
                ierr = nf90_def_var(this%species_group(n), 'u_par', &
                                    NF90_GP, dataset_dims, &
                                    this%u_par_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! E_par
                ierr = nf90_def_var(this%species_group(n), 'E_par', &
                                    NF90_GP, dataset_dims, &
                                    this%E_par_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! E_perp
                ierr = nf90_def_var(this%species_group(n), 'E_perp', &
                                    NF90_GP, dataset_dims, &
                                    this%E_perp_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! Q_par
                ierr = nf90_def_var(this%species_group(n), 'Q_par', &
                                    NF90_GP, dataset_dims, &
                                    this%Q_par_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! Q_perp
                ierr = nf90_def_var(this%species_group(n), 'Q_perp', &
                                    NF90_GP, dataset_dims, &
                                    this%Q_perp_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! K_par
                ierr = nf90_def_var(this%species_group(n), 'K_par', &
                                    NF90_GP, dataset_dims, &
                                    this%K_par_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! K_perp
                ierr = nf90_def_var(this%species_group(n), 'K_perp', &
                                    NF90_GP, dataset_dims, &
                                    this%K_perp_ids(n))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

            end do
            ierr = nf90_enddef(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif
        this%write_counter = 1
    end subroutine

    module subroutine write_mom_2d(this, time, lb_stripped, ub_stripped, &
                                   mom_2d)
        class(mom_2d_file_t), intent(inout) :: this
        real(kind = GP), intent(in) :: time
        integer, intent(in), dimension(5) :: lb_stripped, ub_stripped
        real(kind = GP), intent(in) :: mom_2d(lb_stripped(1) : ub_stripped(1), &
                                              lb_stripped(2) : ub_stripped(2), &
                                              1              : 8, &
                                              lb_stripped(5) : ub_stripped(5))

        integer :: i, j, k, n, ierr, start(3), start_time(1)

        ! Collect the data on the master process
        !$omp parallel shared(this, mom_2d) private(k, i, j, n) &
        !$omp firstprivate(lb_stripped, ub_stripped) default(none)
        do n = 1, this%n_points_sp
            do j = 1, 8
                do k = 1, this%n_points_phi
                    !$omp do schedule(static)
                    do i = 1, this%n_points_RZ
                        if(      i >= lb_stripped(1) &
                           .and. i <= ub_stripped(1) &
                           .and. k >= lb_stripped(2) &
                           .and. k <= ub_stripped(2) &
                           .and. n >= lb_stripped(5) &
                           .and. n <= ub_stripped(5) &
                           ) then
                            this%send_buffer(i, k, j, n) = mom_2d(i, k, j, n)
                        else
                            this%send_buffer(i, k, j, n) = 0.0_GP
                        endif
                    enddo
                    !$omp end do nowait
                enddo
            enddo
        enddo
        !$omp end parallel

        call MPI_Reduce(this%send_buffer, this%recv_buffer, &
                        size(this%send_buffer), MPI_GP, MPI_SUM, &
                        0, this%dcomm_handler%get_comm_phi_sp(), ierr)

        ! Write the data to file
        if(this%dcomm_handler%is_master()) then
            start_time(1) = this%write_counter
            ierr = nf90_put_var(this%file_id, this%time_id, time, &
                                start = start_time)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            start(1) = 1
            start(2) = 1
            start(3) = this%write_counter

            do n = 1, this%n_points_sp
                ! n
                ierr = nf90_put_var(this%species_group(n), &
                                    this%n_ids(n), &
                                    this%recv_buffer(:, :, 1, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! u_par
                ierr = nf90_put_var(this%species_group(n), &
                                    this%u_par_ids(n), &
                                    this%recv_buffer(:, :, 2, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! E_par
                ierr = nf90_put_var(this%species_group(n), &
                                    this%E_par_ids(n), &
                                    this%recv_buffer(:, :, 3, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! E_perp
                ierr = nf90_put_var(this%species_group(n), &
                                    this%E_perp_ids(n), &
                                    this%recv_buffer(:, :, 4, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! Q_par
                ierr = nf90_put_var(this%species_group(n), &
                                    this%Q_par_ids(n), &
                                    this%recv_buffer(:, :, 5, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! Q_perp
                ierr = nf90_put_var(this%species_group(n), &
                                    this%Q_perp_ids(n), &
                                    this%recv_buffer(:, :, 6, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! K_par
                ierr = nf90_put_var(this%species_group(n), &
                                    this%K_par_ids(n), &
                                    this%recv_buffer(:, :, 7, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! K_perp
                ierr = nf90_put_var(this%species_group(n), &
                                    this%K_perp_ids(n), &
                                    this%recv_buffer(:, :, 8, n), &
                                    start = start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            end do
            ierr = nf90_sync(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif

        this%write_counter = this%write_counter + 1
    end subroutine

    module subroutine finalize_mom_2d(this)
        type(mom_2d_file_t), intent(inout) :: this
        integer :: ierr

        if(this%dcomm_handler%is_master()) then
            ierr = nf90_close(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
