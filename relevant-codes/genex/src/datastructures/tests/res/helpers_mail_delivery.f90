module helpers_mail_delivery_m
!! Contains procedures needed for unit testing the mail_delivery_m module
    use mpi
    use genex_fortran_env_m, only : GP, GP_EPS
    use mail_delivery_m, only : deliver_outboxes, finish_delivery
    use dcomm_handler_m, only : dcomm_handler_t
    use math_m, only : almost_equal
    use params_gpu_offload_m, only: get_use_gpu_offload

#ifdef ENABLE_GPU
    use helpers_mail_delivery_gpu_m, only: cbind_test_dimension
#endif

    implicit none

contains

    function test_dimension(comm_world, &
                            n_ranks_phi, &
                            n_ranks_vp, &
                            n_ranks_mu, &
                            n_ranks_sp, &
                            dim_test, &
                            number_of_neighbors) result(res)
        integer, intent(in) :: comm_world
        !! MPI_Comm_World from pFUnit
        integer, intent(in) :: n_ranks_phi
        !! Number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! Number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! Number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! Number of ranks in sp direction
        integer, intent(in) :: dim_test
        !! Dimension to test
        integer, intent(in) :: number_of_neighbors
        !! Number of neighbors
        logical :: res

        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        integer :: i, j
        integer, parameter :: number_of_cells = 4
        integer, parameter :: max_number_of_neighbors = 4
        real(kind=GP) :: inboxes(number_of_cells, max_number_of_neighbors)
        real(kind=GP) :: outboxes(number_of_cells, max_number_of_neighbors)
        integer :: requests_in(max_number_of_neighbors)
        integer :: requests_out(max_number_of_neighbors)
        integer :: partners(max_number_of_neighbors)

        logical :: fine
        integer :: ierr

        res = .true.
        call dcomm_handler%initialize(comm_world, n_ranks_phi, n_ranks_vp, &
                                      n_ranks_mu, n_ranks_sp)
        dcomm_handler_ptr => dcomm_handler

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_test_dimension(dcomm_handler%get_cxx_pointer(), &
                                        dim_test, number_of_neighbors)
            if(ierr /= 0) res = .false.
#endif
        else
            ! setup test data
            do i = 1, number_of_cells
                do j = 1, max_number_of_neighbors
                    inboxes(i, j) = 0.0
                    outboxes(i, j) = real(i * j, kind=GP)
                enddo
            enddo

            call deliver_outboxes(dcomm_handler_ptr, &
                                  number_of_cells, &
                                  number_of_neighbors, &
                                  dim_test, &
                                  inboxes, &
                                  outboxes, &
                                  partners, &
                                  requests_in, &
                                  requests_out)

            call finish_delivery(number_of_neighbors, partners, &
                                 requests_in, requests_out)

            do i = 1, number_of_cells
                do j = 1, number_of_neighbors
                    fine = almost_equal(&
                               outboxes(i, number_of_neighbors - j + 1), &
                               inboxes(i, j), GP_EPS)
                    if(partners(j) /= MPI_PROC_NULL) then
                        if(.not. fine) res = .false.
                    endif
                enddo
            enddo
        endif

    end function

end module
