module mailbox_m
    !! Module implementing the mailbox_t type
    use genex_fortran_env_m, only: GP

    implicit none

    type, public :: mailbox_t
        !! Type containing the buffer data that is sent and received via
        !! MPI communication and information about the mail partners and
        !! the communication requests
        integer :: n_mailbox_cells
        !! Number of cells of the buffer array in the mailbox
        integer, dimension(:), allocatable :: partners
        !! Array of the MPI rank of mail partners or neighbors to exchange data
        integer, dimension(:), allocatable :: requests_in
        !! Array of communication requests from MPI_Irecv for each partners
        integer, dimension(:), allocatable :: requests_out
        !! Array of communication requests from MPI_Isend for each partners
        real(kind=GP), dimension(:,:), allocatable :: inboxes
        !! Buffer array for data received from MPI communation
        real(kind=GP), dimension(:,:), allocatable :: outboxes
        !! Buffer array for data sent for MPI communation
    end type

end module
