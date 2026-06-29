module helpers_collision_operators_m
    ! Submodule that contains helpers for unit tests to test collisions
    use mpi
    use pfunit
    use genex_fortran_env_m, only: GP, GP_EPS
    use logger_m, only: logger_get_debug_channel
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use genex_status_codes_m, only: GENEX_ERR_UTESTS
    use genex_error_handling_m, only: handle_error

    use params_collisions_m, only: get_coll_type
    use params_gpu_offload_m, only: get_use_gpu_offload
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_rho_ref, get_L_ref
    use params_mesh_m, only: get_n_points_sp, get_use_bsg
    use params_bsg_m, only: get_num_bsg_blocks, get_bsg_interp_order

    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use data_storage_m, only: data_storage_5d_t, data_storage_cpu_5d_t
    use data_array_m, only: data_array_4d_t, data_array_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use op_coll_m, only: op_coll_t
    use bsg_operators_m, only: bsg_operators_t
    use profile_container_m, only: profile_container_t

    use test_params_m

    implicit none

contains

    subroutine setup_species_params(comm, rank, is_uniform)
        !! Sets up species parameters masses, charges, and
        !! the numerical parameter temp_scalings.
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        logical, optional, intent(in) :: is_uniform
        !! Determines whether to use uniform parameters (all set to ones)

        logical :: is_uniform_loc
        real(kind=GP), allocatable :: masses(:), charges(:), temp_scalings(:)

        is_uniform_loc = .false.
        if(present(is_uniform)) is_uniform_loc = is_uniform

        if(is_uniform) then
            masses = [1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, &
                      1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
            charges = [-1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, &
                       1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
            temp_scalings = [1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, &
                             1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
        else
            masses = [0.5_GP, 2.3_GP, 3.2_GP, 3.3_GP, &
                      1.4_GP, 2.13_GP, 1.13_GP, 2.35_GP]
            charges = [-1.0_GP, 1.0_GP, 2.0_GP, 3.0_GP, &
                       3.0_GP, 4.0_GP, 2.0_GP, 5.0_GP]
            temp_scalings = [1.33_GP, 2.3_GP, 4.1_GP, 3.1_GP, &
                             2.4_GP, 3.33_GP, 1.3_GP, 1.25_GP]
        end if

        call setup_test_species(comm, rank, &
                                masses=masses, &
                                charges=charges, &
                                temp_scalings=temp_scalings)
    end subroutine

    subroutine setup_mesh(comm, rank, coll_type, dist_type, eq_type)
        !! Sets up mesh parameters grid type, grid size, and quadratures
        !! based on predefined test cases for type of collision, distribution,
        !! and equilibrium.
        use equilibrium_factory_m, only: SLAB, SALPHA, DOMMASCHK

        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=*), intent(in) :: coll_type
        !! Collision type
        character(len=*), intent(in) :: dist_type
        !! Distribution function type
        character(len=*), intent(in) :: eq_type
        !! Equilibrium type

        integer :: n_points_phi, n_points_vp, n_points_mu, equilibrium_type
        real(kind=GP) :: length_vp, length_mu, spacing_RZ
        character(len=:), allocatable :: grid_type_mu, quad_type_vp

        select case(dist_type)
            case("poly2")
                length_vp = 3.0_GP
                length_mu = 9.0_GP
            case("maxw", "corr_maxw", "bi_maxw", "double_maxw")
                length_vp = 8.0_GP
                length_mu = 64.0_GP
            case default
                call handle_error("Unsupported dist_type " // dist_type, &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end select

        n_points_phi = 1
        select case(eq_type)
            case("slab")
                equilibrium_type = SLAB
                spacing_RZ = 16.0_GP
            case("dommaschk")
                equilibrium_type = DOMMASCHK
                spacing_RZ = 0.012_GP
            case("salpha")
                equilibrium_type = SALPHA
                spacing_RZ = 0.095_GP
            case default
                call handle_error("Unsupported eq_type " // eq_type, &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end select

        select case(coll_type)
            case("bgk")
                grid_type_mu = "gauss-laguerre"
                quad_type_vp = "simpson"
                n_points_vp = 64
                n_points_mu = 64
            case("lbd")
                grid_type_mu = "quadratic"
                quad_type_vp = "midpoint"
                n_points_vp = 64
                n_points_mu = 64
            case("lorentz")
                grid_type_mu = "quadratic"
                quad_type_vp = "midpoint"
                n_points_vp = 64
                n_points_mu = 64
            case("fpl")
                grid_type_mu = "quadratic"
                quad_type_vp = "midpoint"
                n_points_vp = 16
                n_points_mu = 16
            case default
                call handle_error("Unsupported coll_type " // coll_type, &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end select

        call setup_test_equi_slab(comm, rank, sol=.false.)
        call setup_test_mesh(comm, rank, &
                             equilibrium_type=equilibrium_type, &
                             spacing_RZ=spacing_RZ, &
                             n_points_phi=n_points_phi, &
                             n_points_vp=n_points_vp, &
                             n_points_mu=n_points_mu, &
                             length_vp=length_vp, &
                             length_mu=length_mu, &
                             quad_type_vp=quad_type_vp, &
                             grid_type_mu=grid_type_mu)
    end subroutine

    subroutine setup_coll(comm, rank, coll_type, op_coll)
        !! Allocates op_coll based on coll_type and sets up coll_type parameter
        use op_coll_m, only: op_coll_t, op_coll_bgk_cpu_t, op_coll_lbd_cpu_t, &
                             op_coll_lorentz_cpu_t, op_coll_fpl_cpu_t
#ifdef ENABLE_GPU
        use op_coll_m, only: op_coll_bgk_gpu_t
#endif

        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=*), intent(in) :: coll_type
        !! Collision type
        class(op_coll_t), allocatable, intent(out) :: op_coll

        call setup_test_coll(comm, rank, coll_type=coll_type)

        if(get_coll_type() == "bgk" .and. .not. get_use_gpu_offload()) then
            allocate(op_coll_bgk_cpu_t :: op_coll)
        else if(get_coll_type() == "lbd") then
            allocate(op_coll_lbd_cpu_t :: op_coll)
        else if(get_coll_type() == "lorentz") then
            allocate(op_coll_lorentz_cpu_t :: op_coll)
        else if(get_coll_type() == "fpl") then
            allocate(op_coll_fpl_cpu_t :: op_coll)
#ifdef ENABLE_GPU
        else if(get_coll_type() == "bgk" .and. get_use_gpu_offload()) then
            allocate(op_coll_bgk_gpu_t :: op_coll)
#endif
        else
            call handle_error("Unsupported coll_type " // coll_type, &
                              GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end if
    end subroutine

    subroutine setup_profile_container(profile_container)
        !! Sets up profile container based on predefined profile parameters.
        use profiles_m, only: profile_t, profile_cbc_cpu_t
        use params_profile_m, only: params_profile_t
        use params_profile_cbc_m, only: params_profile_cbc_t, &
                                        get_params_profile_cbc_dens, &
                                        get_params_profile_cbc_temp

        type(profile_container_t), allocatable, intent(out) :: &
                                                            profile_container(:)

        class(params_profile_t), allocatable :: params_profile_dens
        class(params_profile_t), allocatable :: params_profile_temp
        class(profile_t), allocatable :: profile_dens
        class(profile_t), allocatable :: profile_temp
        integer :: n, n_sp

        n_sp = get_n_points_sp()

        allocate(profile_container(n_sp))
        do n = 1, n_sp
            allocate(params_profile_cbc_t :: params_profile_dens)
            allocate(params_profile_cbc_t :: params_profile_temp)
            allocate(profile_cbc_cpu_t :: profile_dens)
            allocate(profile_cbc_cpu_t :: profile_temp)

            params_profile_dens = get_params_profile_cbc_dens(n)
            params_profile_temp = get_params_profile_cbc_temp(n)

            select type(profile_dens)
                type is(profile_cbc_cpu_t)
                    select type(params_profile_dens)
                        type is(params_profile_cbc_t)
                            call profile_dens%initialize(params_profile_dens)
                    end select
            end select
            select type(profile_temp)
                type is(profile_cbc_cpu_t)
                    select type(params_profile_temp)
                        type is(params_profile_cbc_t)
                            call profile_temp%initialize(params_profile_temp)
                    end select
            end select

            call profile_container(n)%initialize(.true., profile_dens, &
                                                 profile_temp, profile_temp, &
                                                 profile_temp)

            deallocate(params_profile_dens, params_profile_temp, &
                       profile_dens, profile_temp)
        end do
    end subroutine

    subroutine setup_dcomm_handler(comm, n_procs, dcomm_handler)
        !! Sets up communication handler using predefined parallelization
        !! depending on the number of processes used.
        integer, intent(in) :: comm
        integer, intent(in) :: n_procs
        type(dcomm_handler_t), intent(out) :: dcomm_handler

        integer :: n_procs_phi, n_procs_vp, n_procs_mu, n_procs_sp

        select case(n_procs)
            case(1)
                n_procs_phi = 1
                n_procs_vp = 1
                n_procs_mu = 1
                n_procs_sp = 1
            case(2)
                n_procs_phi = 1
                n_procs_vp = 2
                n_procs_mu = 1
                n_procs_sp = 1
            case(4)
                n_procs_phi = 1
                n_procs_vp = 2
                n_procs_mu = 2
                n_procs_sp = 1
            case(8)
                n_procs_phi = 2
                n_procs_vp = 1
                n_procs_mu = 2
                n_procs_sp = 2
            case default
                call handle_error("Unsupported n_procs!", &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end select

        call dcomm_handler%initialize(comm, n_procs_phi, &
                                      n_procs_vp, n_procs_mu, n_procs_sp)
    end subroutine

    subroutine setup_dist_initial(comm, rank, dcomm_handler, mesh, dist_type, &
                                  profile_container, data_storage)
        !! Sets initial distribution function and performs ghost exchange.
        !! Available options for dist_type are:
        !! 1) maxw
        !! 2) bi-maxw
        !! 3) double-maxw,
        !! 4) corr-maxw (maxw * bps / absB),
        !! 5) poly2 (second-order polynomial)
        use dist_initial_container_m, only: dist_initial_container_t
        use dist_initial_m, only: dist_initial_t
        use dist_initial_m, only: dist_initial_maxw_cpu_t, &
                                  dist_initial_bi_maxw_cpu_t, &
                                  dist_initial_double_maxw_cpu_t
        use params_dist_initial_m, only: params_dist_initial_t
        use params_dist_initial_bi_maxw_m, only: &
            params_dist_initial_bi_maxw_t, &
            get_params_dist_initial_bi_maxw
        use params_dist_initial_double_maxw_m, only: &
            params_dist_initial_double_maxw_t, &
            get_params_dist_initial_double_maxw
        use dimensions_m, only: DIM_MU, DIM_VP
        use op_set_initial_condition_m, only: op_set_initial_condition_cpu_t

        integer, intent(in) :: comm
        integer, intent(in) :: rank
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout):: mesh
        character(len=*), intent(in) :: dist_type
        type(profile_container_t), dimension(:), intent(in) :: profile_container
        type(data_storage_cpu_5d_t), intent(inout) :: data_storage

        class(data_array_5d_t), pointer :: da_f
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f
        class(params_dist_initial_t), allocatable :: params_dist
        class(dist_initial_t), allocatable :: dist_initial
        type(dist_initial_container_t), allocatable :: dist_initial_container(:)
        type(op_set_initial_condition_cpu_t), allocatable :: &
                                                        op_set_initial_condition
        type(bsg_operators_t), allocatable :: bsg_op

        integer :: n, n_sp, lb(5), ub(5), lb_stripped(5), ub_stripped(5)

        da_f => data_storage%get_data_pointer()
        f => da_f%get_pointer()
        lb = da_f%get_lbound()
        ub = da_f%get_ubound()
        lb_stripped = da_f%get_lbound_stripped()
        ub_stripped = da_f%get_ubound_stripped()
        n_sp = mesh%size_sp()

        allocate(dist_initial_container(n_sp))
        do n = 1, n_sp
            ! We allocate both dist_initial and params_dist regardless if they
            ! are used. This is to prevent many if-branches.
            select case(dist_type)
                case("maxw", "corr_maxw", "poly2")
                    allocate(dist_initial_maxw_cpu_t :: dist_initial)
                    allocate(params_dist_initial_double_maxw_t :: params_dist)
                case("bi_maxw")
                    allocate(dist_initial_bi_maxw_cpu_t :: dist_initial)
                    allocate(params_dist_initial_bi_maxw_t :: params_dist)
                case("double_maxw")
                    allocate(dist_initial_double_maxw_cpu_t :: dist_initial)
                    allocate(params_dist_initial_double_maxw_t :: params_dist)
                case default
                    call handle_error("Unsupported dist_type " // dist_type, &
                                      GENEX_ERR_UTESTS, __LINE__, __FILE__)
            end select

            select type(params_dist)
                type is(params_dist_initial_bi_maxw_t)
                    params_dist%drift_vpar = 0.27_GP
                    call setup_test_dist_initial(comm, rank, n, params_dist)
                type is(params_dist_initial_double_maxw_t)
                    params_dist%drift_par1  = -0.27_GP
                    params_dist%drift_par2  = -0.54_GP
                    call setup_test_dist_initial(comm, rank, n, params_dist)
            end select

            call dist_initial%initialize(profile_container, n)
            call dist_initial_container(n)%initialize(dist_initial)
            deallocate(params_dist, dist_initial)
        end do

        allocate(bsg_operators_t :: bsg_op)
        if (get_use_bsg()) then
            call bsg_op%initialize(mesh, get_num_bsg_blocks(), &
                                   get_bsg_interp_order())
        else
            call bsg_op%initialize(mesh, 1)
        end if

        allocate(op_set_initial_condition)
        call op_set_initial_condition%initialize(dcomm_handler, mesh, &
                                                 dist_initial_container, bsg_op)
        call op_set_initial_condition%apply(da_f)

        block
            integer :: i, k, l, m, n
            real(kind=GP) :: bps
            real(kind=GP), dimension(:), allocatable :: prefac_bps
            real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu
            real(kind=GP), contiguous, pointer, dimension(:,:) :: absB
            real(kind=GP), contiguous, pointer, dimension(:,:) :: curl_normb_y

            allocate(prefac_bps(n_sp))
            do n = 1, n_sp
                prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                            * get_temp_scaling(n)) &
                                * get_rho_ref() &
                                / (get_charge(n) * get_L_ref())
            end do

            absB         => mesh%get_absB_pointer()
            curl_normb_y => mesh%get_curl_normb_y_pointer()
            vp           => mesh%get_vp_pointer()
            mu           => mesh%get_mu_pointer()

            do n = lb_stripped(5), ub_stripped(5)
            do m = lb_stripped(4), ub_stripped(4)
            do l = lb_stripped(3), ub_stripped(3)
            do k = lb_stripped(2), ub_stripped(2)
            do i = lb_stripped(1), ub_stripped(1)
                bps = absB(i, k) &
                    + prefac_bps(n) * vp(l) * curl_normb_y(i, k)
                if(dist_type == "corr_maxw") then
                    f(i, k, l, m, n) = f(i, k, l, m, n) &
                                     * absB(i, k) / bps
                else if(dist_type == "poly2") then
                    ! Parameters are scaled to approx same magnitude as a maxw
                    ! (to achieve similar absolute errors)
                    f(i, k, l, m, n) = 6.34e-4_GP &
                                     + 12.25e-4_GP * vp(l) &
                                     - 29.83e-4 * vp(l)**2.0_GP &
                                     + 124.14e-4 * mu(m) &
                                     - 13.9e-4 * mu(m)**2.0_GP
                end if
            end do
            end do
            end do
            end do
            end do
        end block

        ! Performs ghost exchange including corner cells
        call data_storage%start_exchange(DIM_VP)
        call data_storage%finish_exchange(DIM_VP)
        call data_storage%start_exchange(DIM_MU)
        call data_storage%finish_exchange(DIM_MU)
    end subroutine

    subroutine calc_coll(dcomm_handler, mesh, op_coll, da_f_in, da_f_out)
        !! Calculates the value of op_coll performed on da_f_in.
        !! Stores the result in da_f_out.
        use op_mom_coll_m, only: op_mom_coll_t, op_mom_coll_cpu_t
#ifdef ENABLE_GPU
        use op_mom_coll_m, only: op_mom_coll_gpu_t
#endif

        type(dcomm_handler_t), target, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout):: mesh
        class(op_coll_t), intent(inout) :: op_coll
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_5d_t), intent(out) :: da_f_out

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_out

        class(op_mom_coll_t), allocatable :: op_mom
        class(data_array_4d_t), allocatable :: da_mom_coll

        integer :: n_sp, lb_stripped(5), ub_stripped(5)

        n_sp = mesh%size_sp()
        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            allocate(op_mom_coll_gpu_t :: op_mom)
#endif
        else
            allocate(op_mom_coll_cpu_t :: op_mom)
        end if
        call op_mom%initialize(dcomm_handler, mesh)

        allocate(da_mom_coll)
        call da_mom_coll%initialize([lb_stripped(1), lb_stripped(2), 1, 1], &
                                    [ub_stripped(1), ub_stripped(2), 3, n_sp])
        call da_f_out%initialize(lb_stripped, ub_stripped)
        call op_mom%apply(da_f_in, da_mom_coll)
        call op_coll%apply(da_f_in, da_mom_coll, da_f_out)
        call da_f_out%update_host()
    end subroutine

    subroutine calc_mom(dcomm_handler, mesh, da_f, da_g, da_moments)
        !! Calculates species-independent-normalized moments of da_g.
        !! Outputs in third index of da_moments.
        !! The following moments are calculated:
        !! 1) Density
        !! 2) Momentum
        !! 3) Energy
        !! 4) H-function w.r.t. da_f
        type(dcomm_handler_t), intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout):: mesh
        class(data_array_5d_t), intent(in) :: da_f
        class(data_array_5d_t), intent(in) :: da_g
        class(data_array_4d_t), intent(inout) :: da_moments

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f, g
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        real(kind=GP), allocatable, dimension(:,:,:,:) :: send_buffer
        real(kind=GP), contiguous, pointer, dimension(:) :: vp, mu, muw, vpw, &
                                                            sqrt_mu, sqrt_muw
        real(kind=GP), contiguous, pointer, dimension(:,:) :: absB, curl_normb_y
        real(kind=GP), allocatable, dimension(:) :: prefac_mom, prefac_energy, &
                                                    prefac_bps

        real(kind=GP) :: bps, area
        integer :: i, j, k, l, m, n, n_sp, ierr
        integer, dimension(5) :: lb_stripped, ub_stripped
        type(op_set_uniform_cpu_t) :: op_set_uniform

        f => da_f%get_readonly_pointer()
        g => da_g%get_readonly_pointer()
        moments => da_moments%get_pointer()
        lb_stripped = da_f%get_lbound_stripped()
        ub_stripped = da_f%get_ubound_stripped()

        n_sp = mesh%size_sp()
        absB => mesh%get_absB_pointer()
        curl_normb_y => mesh%get_curl_normb_y_pointer()
        vp => mesh%get_vp_pointer()
        mu => mesh%get_mu_pointer()
        vpw => mesh%get_vpw_pointer()
        muw => mesh%get_muw_pointer()

        ! Calculate prefactors for moment calculation
        allocate(prefac_bps(n_sp), prefac_mom(n_sp), prefac_energy(n_sp))
        do n = 1, n_sp
            prefac_bps(n) = sqrt(2.0_GP * get_mass(n) &
                                        * get_temp_scaling(n)) &
                          * get_rho_ref() / (get_charge(n) * get_L_ref())
            prefac_mom(n) = sqrt(2.0_GP * get_temp_scaling(n) * get_mass(n))
            prefac_energy(n) = get_temp_scaling(n)
        enddo

        ! Allocate send buffer
        ! Indices: 1) unstructured grid, 2) angle phi, 3) quantity,
        !          4) species (all, not only of mpi task)
        if(.not. allocated(send_buffer)) then
            allocate(send_buffer(lb_stripped(1):ub_stripped(1), &
                                 lb_stripped(2):ub_stripped(2), 4, n_sp))
        end if

        ! First touch initialize and reset buffer
        call op_set_uniform%apply(send_buffer, 0.0_GP)

        ! Dependent on the perpendicular grid type we perform integration in
        ! mu or vperp. This is important for the finite volume implementations
        ! of collision operators, since the quadrature scheme must be the same
        ! in the implementation and in the test. Otherwise we cannot test for
        ! conservation up to machine precision.

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(f, g, vp, mu, absB, vpw, muw, &
        !$omp        prefac_mom, prefac_energy, prefac_bps, &
        !$omp        curl_normb_y, send_buffer) &
        !$omp private(i, k, l, m, n, bps, area)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do simd schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            bps = absB(i, k) + prefac_bps(n) * vp(l) * curl_normb_y(i, k)
            area = vpw(l) * muw(m) * bps * PI

            ! Density
            send_buffer(i, k, 1, n) = send_buffer(i, k, 1, n) &
                                    + area * g(i, k, l, m, n)
            ! Momentum
            send_buffer(i, k, 2, n) = send_buffer(i, k, 2, n) &
                                    + area * vp(l) * g(i, k, l, m, n) &
                                      * prefac_mom(n)
            ! Energy
            send_buffer(i, k, 3, n) = send_buffer(i, k, 3, n) &
                                    + area * (vp(l)**2 + absB(i, k) * mu(m)) &
                                      * g(i, k, l, m, n) * prefac_energy(n)
            ! H-function w.r.t. f
            send_buffer(i, k, 4, n) = send_buffer(i, k, 4, n) &
                                    - area * g(i, k, l, m, n) &
                                      * log(abs(f(i, k, l, m, n)) + GP_EPS)
        end do
        !$omp end do simd nowait
        end do
        end do
        end do
        end do
        !$omp end parallel

        ! Collect results from all mpi processes
        call MPI_Allreduce(send_buffer, moments, size(send_buffer), MPI_GP, &
                           MPI_SUM, dcomm_handler%get_comm_vp_mu_sp(), ierr)
    end subroutine

end module
