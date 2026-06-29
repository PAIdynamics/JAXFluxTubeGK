module mail_delivery_m
    !! Module containing the implementation of a general data exchange within a
    !! communicator
    use dcomm_handler_m, only: dcomm_handler_t
    use genex_fortran_env_m, only: GP, MPI_GP

    implicit none

    interface
        module subroutine deliver_outboxes(dcomm_handler, &
                                           number_of_cells, &
                                           number_of_mail_partners, &
                                           exchange_dimension, &
                                           inboxes, &
                                           outboxes, &
                                           neighbors, &
                                           requests_in, &
                                           requests_out)
            !! Exchange the information stored in outboxes over the
            !! selected dimension with the specified number of neighbors
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            integer, intent(in) :: number_of_cells
            !! Number of cells of the buffer array in the mailbox
            integer, intent(in) :: number_of_mail_partners
            !! Number of mail partners or neighbors to exchange data in mailbox
            integer, intent(in) :: exchange_dimension
            !! Dimension to exchange over
            real(kind=GP), dimension(number_of_cells, &
                                     number_of_mail_partners), &
                           intent(in) :: inboxes
            !! Buffer array for data received from MPI communation
            real(kind=GP), dimension(number_of_cells, &
                                     number_of_mail_partners), &
                            intent(inout) :: outboxes
            !! Buffer array for data sent for MPI communation
            integer, dimension(number_of_mail_partners), intent(inout) :: &
                neighbors
            !! MPI rank array of mail partners/neighbors to exchange data
            integer, dimension(number_of_mail_partners), &
                     intent(out) :: requests_in
            !! Array of communication requests from MPI_Irecv for each partners
            integer, dimension(number_of_mail_partners), &
                     intent(out) :: requests_out
            !! Array of communication requests from MPI_Isend for each partners
        end subroutine

        module subroutine finish_delivery(number_of_mail_partners, &
                                          neighbors, &
                                          requests_in, &
                                          requests_out)
            !! Finish the exchange of information stored in the outboxes
            integer, intent(in) :: number_of_mail_partners
            !! Number of mail partners or neighbors to exchange data in mailbox
            integer, dimension(number_of_mail_partners), &
                intent(in) :: neighbors
            !! MPI rank array of mail partners/neighbors to exchange data
            integer, dimension(number_of_mail_partners), &
                     intent(inout) :: requests_in
            !! Array of communication requests from MPI_Irecv for each partners
            integer, dimension(number_of_mail_partners), &
                     intent(inout) :: requests_out
            !! Array of communication requests from MPI_Isend for each partners
        end subroutine
    end interface

end module
