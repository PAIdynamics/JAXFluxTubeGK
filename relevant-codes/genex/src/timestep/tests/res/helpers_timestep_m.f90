module helpers_timestep_m
    !! Contains helper functions for the timestepping schemes
    use mpi
    use pfunit
    use, intrinsic :: iso_fortran_env
    use math_m, only: almost_equal
    use genex_fortran_env_m, only: GP, GP_EPS
    use genex_status_codes_m, only: GENEX_SUCCESS
    use logger_m, only: logger_get_debug_channel
    use dcomm_handler_m, only: dcomm_handler_t
    use state_vector_m, only: state_vector_t
    use timestep_m, only: timestep_t, euler_t, rk4_t, strang_splitting_t, &
                          pirock_t
    use mesh_5d_m, only: mesh_5d_t
    use params_gpu_offload_m, only: get_use_gpu_offload
    ! Unit test helpers
    use test_params_m, only: setup_test_mesh, &
                             setup_test_neutrals_config, &
                             setup_test_time_split
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB

    implicit none

contains

    subroutine test_timestep_core(this, timestep_scheme, dt, tol, passed)
        !! Tests a time stepping scheme
        class(MpiTestMethod), intent(inout) :: this
        character(len=*), intent(in) :: timestep_scheme
        !! Time stepping scheme to test
        real(kind=GP), intent(in) :: dt
        !! Time step
        real(kind=GP), intent(in) :: tol
        !! Absolute test tolerance
        logical, intent(out) :: passed
        !! Result of the test, .true. if passed

        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        type(dcomm_handler_t), target :: dcomm_handler
        type(state_vector_t), pointer:: state
        class(timestep_t), target, allocatable :: timestep
        type(mesh_5d_t), target, allocatable :: mesh
        real(kind=GP), pointer, dimension(:,:,:,:,:) :: res
        integer :: t, ierr, lb_stripped(5)
        real(kind=GP) :: expected, resulted
        integer :: n_t = 10
        logical :: all_fine
        integer :: n_procs, comm, rank

        passed = .true.
        n_procs = this%getNumProcesses()
        comm = this%getMpiCommunicator()
        rank = this%getProcessRank()

        ! Select time stepping scheme for this test
        select case(timestep_scheme)
            case("euler")
                allocate(euler_t :: timestep)
            case("rk4")
                allocate(rk4_t :: timestep)
            case("strang")
                allocate(strang_splitting_t :: timestep)
            case("strang-rkc")
                allocate(strang_splitting_t :: timestep)
                call setup_test_time_split(this%getMpiCommunicator(), &
                                           this%getProcessRank(), &
                                           time_scheme_collisions="rkc", &
                                           time_scheme_neutrals="rkc")
            case("pirock")
                allocate(pirock_t :: timestep)
        end select

        call setup_test_mesh(comm, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=8.0_GP, &
                             n_points_phi=2, &
                             n_points_vp=2, &
                             n_points_mu=2, &
                             quad_type_vp="trapezoidal")

        dcomm_handler_ptr => dcomm_handler
        if(n_procs==1) then
            call dcomm_handler_ptr%initialize(comm, 1, 1, 1, 1)
        elseif(n_procs==8) then
            call dcomm_handler_ptr%initialize(comm, 2, 2, 2, 1)
        endif

        allocate(mesh)
        call mesh%initialize(dcomm_handler_ptr)

        call setup_test_neutrals_config(comm, &
                                        rank, &
                                        neutrals_evolve_type="dummy", &
                                        neutrals_coupling_type="dummy")

        ! Debug log test announcement
        if(dcomm_handler_ptr%is_master()) &
            write(logger_get_debug_channel(), "(A,I1,A,ES9.2)") &
            "Test timestep "//timestep_scheme//" (", &
            n_procs, "procs), tolerance = ", tol

        call timestep%initialize(dcomm_handler_ptr, mesh, dt, "./", &
                                 run_test=.true.)
        state => timestep%get_state_pointer()
        call state%set_uniform(1.0_GP)

        do t = 1, n_t
            call timestep%step(ierr)
            passed = almost_equal(timestep%get_t(), t*dt, 10*GP_EPS)
        enddo
        passed = (ierr == GENEX_SUCCESS)

        call state%update_host_dist_func()
        res => state%dist_func%get_pointer()
        lb_stripped = state%dist_func%lbound()
        expected = exp(1.0_GP - (4.0_GP * n_t * dt) - (n_t * dt)**2 - &
                       cos(n_t * dt) + sin(n_t * dt))
        resulted = res(lb_stripped(1), lb_stripped(2), lb_stripped(3), &
                       lb_stripped(4), lb_stripped(5))
        passed = almost_equal(resulted, expected, tol)

        deallocate(mesh, timestep)
    end subroutine

end module
