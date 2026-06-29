submodule (data_storage_m) data_storage_cpu_5d_s
    use mpi, only: MPI_PROC_NULL
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_DATA_STRUCTURE
    use mail_delivery_m, only: deliver_outboxes, finish_delivery
    use profiler_m, only: profiler_start, profiler_stop
    use dimensions_m, only: DIM_VP, DIM_MU, DIM_PHI

    implicit none

contains

    module subroutine initialize_cpu_5d(this, dcomm_handler, init_value)
        class(data_storage_cpu_5d_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        real(kind=GP), optional, intent(in) :: init_value

        integer :: ex_dims(this%n_ex_dims)
        integer :: ex_dim, i, idim, send_width, number_of_elements

        call this%initialize_parent(dcomm_handler, dimensions=5)

        allocate(this%data_array)
        call this%data_array%initialize(this%lb, this%ub, &
                                        this%lb_stripped, this%ub_stripped, &
                                        val=init_value)
        this%storage => this%data_array%get_pointer()

        ex_dims = [this%dim_permut(DIM_PHI), &
                   this%dim_permut(DIM_VP), &
                   this%dim_permut(DIM_MU)]

        do i = 1, this%n_ex_dims
            ex_dim = ex_dims(i)
            if(this%number_of_mail_partners(ex_dim) == 2) then
                ! Every mail partner gets the complete number of ghost cells
                send_width = this%number_of_ghost_cells(ex_dim)
            else
                ! Every mail partner gets exactly one ghost cell
                send_width = 1
            endif

            if(ex_dim == this%dim_permut(DIM_PHI) &
               .or. ex_dim == this%dim_permut(DIM_VP) &
               .or. ex_dim == this%dim_permut(DIM_MU)) then
                ! Compute the number of data to exchange in each dimension,
                ! excluding ex_dim
                number_of_elements = 1
                do idim = 1, 5
                    if(this%dim_permut(idim) == ex_dim) cycle
                    number_of_elements = number_of_elements &
                       * (this%number_of_data_cells(this%dim_permut(idim)) &
                       + 2 * this%number_of_ghost_cells(this%dim_permut(idim)) &
                         * this%has_edge_ghosts(ex_dim, this%dim_permut(idim)))
                enddo
                this%mailboxes(i)%n_mailbox_cells = send_width &
                                                  * number_of_elements
            else
                call handle_error("Selected Selected exchange dimension &
                                  &is not implemented!", &
                                  GENEX_ERR_DATA_STRUCTURE, __LINE__, __FILE__)
            end if
            ! Allocate storage of the correct size
            allocate(this%mailboxes(i)%inboxes( &
                     this%mailboxes(i)%n_mailbox_cells, &
                     this%number_of_mail_partners(ex_dim)))
            allocate(this%mailboxes(i)%outboxes( &
                     this%mailboxes(i)%n_mailbox_cells, &
                     this%number_of_mail_partners(ex_dim)))
            allocate(this%mailboxes(i)%requests_in( &
                     this%number_of_mail_partners(ex_dim)))
            allocate(this%mailboxes(i)%requests_out( &
                     this%number_of_mail_partners(ex_dim)))
            allocate(this%mailboxes(i)%partners( &
                     this%number_of_mail_partners(ex_dim)))
        enddo
    end subroutine

    subroutine pack_5d(lb_arr_in, arr_in, arr_out, lb, ub, ind)
        !! Packs the elements stored in the part of arr_in specified by lb and
        !! ub into the part of arr_out specified by ind
        integer, dimension(5), intent(in) :: lb_arr_in
        real(kind=GP), dimension(lb_arr_in(1):, lb_arr_in(2):, &
                                 lb_arr_in(3):, lb_arr_in(4):, &
                                 lb_arr_in(5):), intent(in) :: arr_in
        real(kind=GP), dimension(:,:), intent(out) :: arr_out
        integer, dimension(5), intent(in) :: lb
        integer, dimension(5), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(5), i, k, l, m, n, ctr

        shp(1) = (ub(1) - lb(1) + 1)
        do i = 2, 5
            shp(i) = shp(i - 1) * (ub(i) - lb(i) + 1)
        enddo
        ! NOTE: The OpenMP code will only execute in full speed if
        !       dim_permut(1) = DIM_RZ. In the future, if different dim_permuts
        !       are actually used, one has to implement a branch cut with a
        !       specific OpenMP version for a specific dim_permut here and in
        !       unpack_5d
        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k, l, m ,n) shared(arr_out, arr_in)
        do n = lb(5), ub(5)
            do m = lb(4), ub(4)
                do l = lb(3), ub(3)
                    do k = lb(2), ub(2)
                        !$omp do schedule(static)
                        do i = lb(1), ub(1)
                            ctr =   (n - lb(5)) * shp(4) &
                                  + (m - lb(4)) * shp(3) &
                                  + (l - lb(3)) * shp(2) &
                                  + (k - lb(2)) * shp(1) &
                                  +  i - lb(1) + 1
                            arr_out(ctr, ind) = arr_in(i, k, l, m, n)
                        enddo
                        !$omp enddo
                    enddo
                enddo
            enddo
        enddo
        !$omp end parallel
    end subroutine

    subroutine unpack_5d(arr_in, lb_arr_out, arr_out, lb, ub, ind)
        !! Unpacks the elements stored in the part of arr_in specified by ind
        !! into the part of arr_out specified by lb and ub
        real(kind=GP), dimension(:,:), intent(in) :: arr_in
        integer, dimension(5), intent(in) :: lb_arr_out
        real(kind=GP), dimension(lb_arr_out(1):, lb_arr_out(2):, &
                                 lb_arr_out(3):, lb_arr_out(4):, &
                                 lb_arr_out(5):), intent(out) :: arr_out
        integer, dimension(5), intent(in) :: lb
        integer, dimension(5), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(5), i, k, l, m, n, ctr

        shp(1) = (ub(1) - lb(1) + 1)
        do i = 2, 5
            shp(i) = shp(i - 1) * (ub(i) - lb(i) + 1)
        enddo
        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k, l, m ,n) shared(arr_out, arr_in)
        do n = lb(5), ub(5)
            do m = lb(4), ub(4)
                do l = lb(3), ub(3)
                    do k = lb(2), ub(2)
                        !$omp do schedule(static)
                        do i = lb(1), ub(1)
                            ctr =   (n - lb(5)) * shp(4) &
                                  + (m - lb(4)) * shp(3) &
                                  + (l - lb(3)) * shp(2) &
                                  + (k - lb(2)) * shp(1) &
                                  +  i - lb(1) + 1
                            arr_out(i, k, l, m, n) = arr_in(ctr, ind)
                        enddo
                        !$omp enddo
                    enddo
                enddo
            enddo
        enddo
        !$omp end parallel
    end subroutine

    module subroutine start_exchange_cpu_5d(this, ex_dim)
        class(data_storage_cpu_5d_t), intent(inout) :: this
        integer, intent(in) :: ex_dim
        integer :: i, mb_ind
        integer :: lb_pack(5), ub_pack(5)
        integer :: ierr

        mb_ind = this%get_mailbox_index(ex_dim)

        call profiler_start("pack_5d", ierr)
        ! Set generic lower and upper bounds for the data to pack
        do i = 1, 5
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
            if(this%has_edge_ghosts(ex_dim, i) == 1) then
                lb_pack(i) = this%lb(i)
                ub_pack(i) = this%ub(i)
            endif
        enddo

        if(this%number_of_mail_partners(ex_dim) == 4) then
            ! Set specific bounds for the exchange dimension in the case we
            ! have four mail partners
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim)
            ! Pack
            do i = 1, 4
                call pack_5d(this%lb, this%storage, &
                             this%mailboxes(mb_ind)%outboxes, &
                             lb_pack, ub_pack, i)
            end do
        else
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim) &
                + this%number_of_ghost_cells(ex_dim) - 1
            call pack_5d(this%lb, this%storage, &
                         this%mailboxes(mb_ind)%outboxes, &
                         lb_pack, ub_pack, 1)
            lb_pack(ex_dim) = this%ub_stripped(ex_dim) &
                - this%number_of_ghost_cells(ex_dim) + 1
            ub_pack(ex_dim) = this%ub_stripped(ex_dim)
            call pack_5d(this%lb, this%storage, &
                         this%mailboxes(mb_ind)%outboxes, &
                         lb_pack, ub_pack, 2)
        end if
        call profiler_stop("pack_5d", ierr)
        call profiler_start("send_5d", ierr)
        call deliver_outboxes(this%dcomm_handler, &
                              this%mailboxes(mb_ind)%n_mailbox_cells, &
                              this%number_of_mail_partners(ex_dim), &
                              this%get_physical_dimension(ex_dim), &
                              this%mailboxes(mb_ind)%inboxes, &
                              this%mailboxes(mb_ind)%outboxes, &
                              this%mailboxes(mb_ind)%partners, &
                              this%mailboxes(mb_ind)%requests_in, &
                              this%mailboxes(mb_ind)%requests_out)
        call profiler_stop("send_5d", ierr)

    end subroutine

    module subroutine finish_exchange_cpu_5d(this, ex_dim)
        class(data_storage_cpu_5d_t), intent(inout) :: this
        integer, intent(in) :: ex_dim
        integer :: i, mb_ind
        integer :: lb_pack(5), ub_pack(5)
        integer :: ghost_indices(4)
        integer :: ierr

        call profiler_start("receive_5d", ierr)

        mb_ind = this%get_mailbox_index(ex_dim)
        call finish_delivery(this%number_of_mail_partners(ex_dim), &
                             this%mailboxes(mb_ind)%partners, &
                             this%mailboxes(mb_ind)%requests_in, &
                             this%mailboxes(mb_ind)%requests_out)

        call profiler_stop("receive_5d", ierr)
        call profiler_start("unpack_5d", ierr)

        ! Set generic lower and upper bounds for the data to unpack
        do i = 1, 5
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
            if(this%has_edge_ghosts(ex_dim, i) == 1) then
                lb_pack(i) = this%lb(i)
                ub_pack(i) = this%ub(i)
            endif
        enddo

        if(this%number_of_mail_partners(ex_dim) == 4) then
            ghost_indices(1) = this%lb(ex_dim)
            ghost_indices(2) = this%lb_stripped(ex_dim) - 1
            ghost_indices(3) = this%ub_stripped(ex_dim) + 1
            ghost_indices(4) = this%ub(ex_dim)

            do i = 1, 4
                ! Unpack if the mail partner exists
                if(this%mailboxes(mb_ind)%partners(i) /= MPI_PROC_NULL) then
                    lb_pack(ex_dim) = ghost_indices(i)
                    ub_pack(ex_dim) = ghost_indices(i)
                    call unpack_5d(this%mailboxes(mb_ind)%inboxes, &
                                   this%lb, this%storage, &
                                   lb_pack, ub_pack, i)
                endif
            end do
        else
            ! Unpack if the left mail partner exists
            if(this%mailboxes(mb_ind)%partners(1) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%lb(ex_dim)
                ub_pack(ex_dim) = this%lb_stripped(ex_dim) - 1
                call unpack_5d(this%mailboxes(mb_ind)%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 1)
            endif
            ! Unpack if the right mail partner exists
            if(this%mailboxes(mb_ind)%partners(2) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%ub_stripped(ex_dim) + 1
                ub_pack(ex_dim) = this%ub(ex_dim)
                call unpack_5d(this%mailboxes(mb_ind)%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 2)
            endif
        endif
        call profiler_stop("unpack_5d", ierr)

    end subroutine finish_exchange_cpu_5d

end submodule
