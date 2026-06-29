submodule (mail_delivery_m) mail_delivery_s
    use mpi
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_MAIL_DELIVERY

    implicit none

contains

    module subroutine deliver_outboxes(dcomm_handler, &
                                       number_of_cells, &
                                       number_of_mail_partners, &
                                       exchange_dimension, &
                                       inboxes, &
                                       outboxes, &
                                       neighbors, &
                                       requests_in, &
                                       requests_out)
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        integer, intent(in) :: number_of_cells
        integer, intent(in) :: number_of_mail_partners
        integer, intent(in) :: exchange_dimension
        real(kind=GP), dimension(number_of_cells, number_of_mail_partners), &
            intent(in) :: inboxes
        real(kind=GP), dimension(number_of_cells, number_of_mail_partners), &
            intent(inout) :: outboxes
        integer, dimension(number_of_mail_partners), intent(inout) :: neighbors
        integer, dimension(number_of_mail_partners), intent(out) :: requests_in
        integer, dimension(number_of_mail_partners), intent(out) :: requests_out

        integer :: comm, ierr, idx, recv_idx

        requests_in = MPI_REQUEST_NULL
        requests_out = MPI_REQUEST_NULL
        ! Select communicator
        select case(exchange_dimension)
        case(2)
            comm = dcomm_handler%get_comm_phi()
        case(3)
            comm = dcomm_handler%get_comm_vp()
        case(4)
            comm = dcomm_handler%get_comm_mu()
        case default
            call handle_error("Boundary exchange over selected dimension &
                              &is not supported!", &
                              GENEX_ERR_MAIL_DELIVERY, __LINE__, __FILE__)
        end select

        ! Find the neighbor ranks and the number of neighbors
        if(number_of_mail_partners == 2) then
            call mpi_cart_shift(comm, 0, 1, neighbors(1), &
                                neighbors(2), ierr)
        else if(number_of_mail_partners == 4) then
            call mpi_cart_shift(comm, 0, 1, neighbors(2), &
                                neighbors(3), ierr)
            call mpi_cart_shift(comm, 0, 2, neighbors(1), &
                                neighbors(4), ierr)
        else
            call handle_error("Number of mail partners was not 2 or 4!", &
                              GENEX_ERR_MAIL_DELIVERY, __LINE__, __FILE__)
        endif

        do idx = 1, number_of_mail_partners
            if(neighbors(idx) /= MPI_PROC_NULL) then
                ! Neighbor exists
                call mpi_isend(outboxes(:, idx), size(outboxes(:,idx)), &
                               MPI_GP, neighbors(idx), idx, comm, &
                               requests_out(idx), ierr)
            endif
        enddo
        do idx = 1, number_of_mail_partners
            recv_idx = number_of_mail_partners - idx + 1
            if(neighbors(recv_idx) /= MPI_PROC_NULL) then
                ! Neighbor exists
                call mpi_irecv(inboxes(:, recv_idx), size(inboxes(:,idx)), &
                               MPI_GP, neighbors(recv_idx), idx, comm, &
                               requests_in(recv_idx), ierr)
            endif
        end do

    end subroutine

    module subroutine finish_delivery(number_of_mail_partners, &
                                      neighbors, &
                                      requests_in, &
                                      requests_out)
        integer, intent(in) :: number_of_mail_partners
        integer, dimension(number_of_mail_partners), intent(in) :: neighbors
        integer, dimension(number_of_mail_partners), &
                 intent(inout) :: requests_in
        integer, dimension(number_of_mail_partners), &
                 intent(inout) :: requests_out
        integer :: ierr, idx, recv_idx

        do idx = 1, number_of_mail_partners
            recv_idx = number_of_mail_partners - idx + 1
            ! Check whether the neighbor we receive from exists
            if(neighbors(recv_idx) /= MPI_PROC_NULL) then
                call mpi_wait(requests_in(recv_idx), MPI_STATUS_IGNORE, ierr)
            endif
            ! Check whether the neighbor we might have send to exists
            if(neighbors(idx) /= MPI_PROC_NULL) then
                call mpi_wait(requests_out(idx), MPI_STATUS_IGNORE, ierr)
            endif
        enddo

    end subroutine

end submodule
