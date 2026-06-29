module helpers_dcomm_handler_m
    !! Contains procedures needed for unit testing the dcomm_handler_t type
    use mpi
    use dcomm_handler_m, only: dcomm_handler_t
    use params_gpu_offload_m, only: get_use_gpu_offload

#ifdef ENABLE_GPU
    use helpers_dcomm_handler_gpu_m, only: cbind_dcomm_handler_test_rank_config
#endif

    implicit none

contains
    function test_communicator(comm, initial_value, expected_value) result(res)
        !! Tests the given communicator by performing an mpi sum reduction.
        !! Returns true if the test is successful and false otherwise
        integer, intent(in) :: comm
        !! Communicator to test
        integer, intent(in) :: initial_value
        !! Initial value for the executing rank
        integer, intent(in) :: expected_value
        !! Expected value of the sum reduction
        integer :: local_sum, global_sum, ierr
        logical :: res

        call MPI_AllReduce(initial_value, global_sum, 1, MPI_INTEGER, &
                           MPI_SUM, comm, ierr)
        if(ierr /= 0) then
            ! Test failed due to mpi error
            res = .false.
        endif
        if(global_sum == expected_value) then
            res = .true.
        else
            res = .false.
        endif
    end function

    function test_rank_configuration(comm_world, &
                                     rank, &
                                     n_procs, &
                                     n_procs_phi, &
                                     n_procs_vp, &
                                     n_procs_mu, &
                                     n_procs_sp) result(res)
        !! Wrapper routine to test the dcomm_handler_t type for a given rank
        !! config. Returns true if the test is successful and false otherwise.
        integer, intent(in) :: comm_world
        !! MPI_Comm_World from pFUnit
        integer, intent(in) :: rank
        !! Currenk rank
        integer, intent(in) :: n_procs
        !! Number of procs
        integer, intent(in) :: n_procs_phi
        !! Number of procs in phi direction
        integer, intent(in) :: n_procs_vp
        !! Number of procs in vp direction
        integer, intent(in) :: n_procs_mu
        !! Number of procs in mu direction
        integer, intent(in) :: n_procs_sp
        !! Number of procs in sp direction
        logical :: res

        type(dcomm_handler_t) :: dcomm_handler
        integer :: comm, ierr, initial_value, expected_value
        logical :: fine

        res = .true.
        call dcomm_handler%initialize(comm_world, n_procs_phi, n_procs_vp, &
                                      n_procs_mu, n_procs_sp)

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_dcomm_handler_test_rank_config( &
                   dcomm_handler%get_cxx_pointer())
            if(ierr /= 0) res = .false.
#endif
        else

            ! test complete topology
            comm = dcomm_handler%get_comm_cart()
            initial_value = rank
            expected_value = n_procs * (n_procs - 1) / 2
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test phi communicator
            comm = dcomm_handler%get_comm_phi()
            initial_value = 2
            expected_value = 2 * n_procs_phi
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test vp communicator
            comm = dcomm_handler%get_comm_vp()
            initial_value = 2
            expected_value = 2 * n_procs_vp
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test mu communicator
            comm = dcomm_handler%get_comm_mu()
            initial_value = 2
            expected_value = 2 * n_procs_mu
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test sp communicator
            comm = dcomm_handler%get_comm_sp()
            initial_value = 2
            expected_value = 2 * n_procs_sp
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test vp_mu communicator
            comm = dcomm_handler%get_comm_vp_mu()
            initial_value = 2
            expected_value = 2 * n_procs_vp * n_procs_mu
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test vp_mu_sp communicator
            comm = dcomm_handler%get_comm_vp_mu_sp()
            initial_value = 2
            expected_value = 2 * n_procs_vp * n_procs_mu * n_procs_sp
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test phi_vp_mu communicator
            comm = dcomm_handler%get_comm_phi_vp_mu()
            initial_value = 2
            expected_value = 2 * n_procs_phi * n_procs_vp * n_procs_mu
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test mu_sp communicator
            comm = dcomm_handler%get_comm_mu_sp()
            initial_value = 2
            expected_value = 2 * n_procs_mu * n_procs_sp
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

            ! test phi_mu_sp communicator
            comm = dcomm_handler%get_comm_phi_mu_sp()
            initial_value = 2
            expected_value = 2 * n_procs_phi * n_procs_mu * n_procs_sp
            fine = test_communicator(comm, initial_value, expected_value)
            if(.not. fine) res = .false.

        endif

    end function
end module
