submodule(mesh_5d_m) mesh_5d_s
    use genex_fortran_env_m, only: SP, DP, GP, GP_EPS
    use genex_status_codes_m, only: GENEX_ERR_MESH, GENEX_WRN_MESH, &
                                    GENEX_ERR_BSG
    use genex_error_handling_m, only: handle_error, error_info_t
    use math_m, only: PI, almost_equal
    use params_m, only: get_parameter_file
    use params_mesh_m, only: get_equilibrium_type, get_spacing_RZ, &
                             get_n_points_phi, get_n_points_vp, &
                             get_n_points_mu, &
                             get_length_mu, get_length_vp, &
                             get_quad_type_vp, get_grid_type_mu, &
                             get_n_levels, get_use_vspectral, &
                             get_only_first_field_period, &
                             get_use_bsg, get_reorder_size, &
                             get_extend_beyond_wall
    use params_bsg_m, only: get_num_bsg_blocks, get_vplen_markers, &
                            get_radial_markers
    use params_numerical_scheme_m, only: get_buf_zone_size, &
                                         get_buf_zone_size_axis
    use params_field_solve_m, only: get_maxiter, get_smoother, get_restol_zero
    use params_devtools_m, only: get_parallax_dbgout
    use type_converters_m, only: string
    use diag_files_m, only: mesh_file_t
    use params_gpu_offload_m, only: get_use_gpu_offload, &
                                    get_parallax_gpu_offload_backend, &
                                    get_use_parallax_gpu_data_explicit, &
                                    get_swap_mesh_members, &
                                    PARALLAX_BACKEND_CPU, PARALLAX_BACKEND_GPU
    use params_diagnostics_m, only: get_diagnose_tpc, &
                                    get_polar_n_theta, get_polar_n_rho

    use profiler_m, only: profiler_start, profiler_stop

    ! From PARALLAX
    use equilibrium_factory_m, only: create_equilibrium
    use fieldline_tracer_m, only: trace
    use descriptors_m, only: DISTRICT_PRIVFLUX, DISTRICT_SOL, DISTRICT_WALL, &
                             DISTRICT_DOME, DISTRICT_OUT, &
                             BND_TYPE_DIRICHLET_ZERO
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cpu_t
    use helmholtz_solver_factory_m, only: helmholtz_solver_factory, &
                                          parameters_helmholtz_solver_factory

#ifdef ENABLE_PARALLAX_GPU
    ! From PARALLAX
    use helmholtz_solver_m, only: helmholtz_solver_mgmres_cxx_t
#endif

    implicit none

contains

    module subroutine initialize(this, dcomm_handler, read_mesh_from_file, &
                                 mesh_filename)
        class(mesh_5d_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        logical, optional, intent(in) :: read_mesh_from_file
        character(len=*), optional, intent(in) :: mesh_filename

        real(kind=GP), pointer, contiguous, dimension(:) :: phi
        ! Local storage for phi grid
        real(kind=GP) :: spacing_RZ
        ! Mesh spacing requires a local, because of intent(inout) attribute in
        ! PARALLAX
        integer :: n_points_phi
        ! Number of poloidal planes (size of phi dimension)
        real(kind=GP) :: phi_max
        ! Maximum value of phi grid
        integer :: num_field_periods
        ! Number of field periods in equilibrium
        logical :: read_mesh_from_file_local
        ! Local variable for mesh read switch
        integer :: multigrid_levels
        ! Number of multigrid levels
        integer, dimension(:), allocatable :: multigrid_reorder_size
        ! Array with reorder instructions for the multigrid used in the field
        ! solvers
        integer :: k, nb
        ! Loop index
        real(kind=GP) :: vp_bsg_length
        ! Length of the vp grid
        integer :: ierr

        read_mesh_from_file_local = .false.
        if(present(read_mesh_from_file)) then
            read_mesh_from_file_local = read_mesh_from_file
        endif

        spacing_RZ = get_spacing_RZ()
        n_points_phi = get_n_points_phi()
        multigrid_levels = get_n_levels()

        this%dcomm_handler => dcomm_handler

        this%equilibrium_type_storage = get_equilibrium_type()

        call profiler_start("initialize_equilibrium", ierr, set_path=.true.)

        ! Initialize equilibrium from PARALLAX
        if(.not. is_equilibrium_initialized) then
            if(trim(get_parameter_file()) /= "") then
                call create_equilibrium(equilibrium_instance, &
                                        this%equilibrium_type_storage, &
                                        get_parameter_file(), &
                                        dbgout=get_parallax_dbgout())
            else
                ! No parameter file has been provided, initialize to default
                ! value for unit testing
                call create_equilibrium(equilibrium_instance, &
                                        this%equilibrium_type_storage, &
                                        dbgout=get_parallax_dbgout())
            endif
            is_equilibrium_initialized = .true.
        else
            call handle_error("Attempted to create more than one instance of &
                              &the mesh!", GENEX_ERR_MESH, &
                              __LINE__, __FILE__)
        endif

        call profiler_stop("initialize_equilibrium", ierr, set_path=.true.)

        ! Initialize the grids in vp and mu
        allocate(this%bsg_instance)
        if(get_use_bsg()) then
            if (get_use_vspectral() .or. get_use_gpu_offload()) then
                call handle_error("BSG is not supported with vspectral and &
                                  &GPU offload", GENEX_ERR_MESH, &
                                  __LINE__, __FILE__)
            end if

            allocate(this%vp_grid(get_num_bsg_blocks()), this%mu_grid)
            do nb = 1, get_num_bsg_blocks()
                ! For BSG case, vp_grid is only initialized here and the actual
                ! construction is done after obtaining RZ grid information
                call this%vp_grid(nb)%initialize( &
                                               get_n_points_vp(), &
                                               is_vspectral=get_use_vspectral())
            end do
            call this%bsg_instance%initialize(get_num_bsg_blocks(), &
                                              get_radial_markers())
            call this%bsg_instance%init_vp_grid(get_n_points_vp(), &
                                                get_length_vp(), &
                                                get_vplen_markers())
        else
            ! Non BSG case
            call this%bsg_instance%initialize(1)
            allocate(this%vp_grid(1), this%mu_grid)
            call this%vp_grid(1)%initialize(get_n_points_vp(), &
                                            is_vspectral=get_use_vspectral())
            ! Construct SG here as RZ grid information is not needed here
            call this%vp_grid(1)%construct(get_length_vp(), get_quad_type_vp())
        end if

        call this%mu_grid%initialize(get_length_mu(), get_n_points_mu(), &
                                     get_grid_type_mu(), &
                                     is_vspectral=get_use_vspectral())

        ! Initialize the grid in phi, either the full torus or only the
        ! first field period
        allocate(this%phi_grid)

        phi_max = 2.0_GP * PI
        if(get_only_first_field_period()) then
            select type(equilibrium_instance)
            type is(dommaschk_t)
                num_field_periods = equilibrium_instance%get_num_field_periods()
                phi_max = phi_max / num_field_periods
            class default
                call handle_error("Specified only_first_field_period for &
                                  &an invalid equilibrium type! &
                                  &(option ignored)", &
                                  GENEX_WRN_MESH, __LINE__, __FILE__)
            end select
        endif

        ! Initialize moment spectral weight
        if(get_use_vspectral()) then
            call this%initialize_mom_weights_vspec()
        endif

        call this%phi_grid%initialize(phi_max, n_points_phi)
        phi => this%phi_grid%get_pointer()

        ! Get the amount of phi planes required for initialization
        ! NOTE: Currently we have to build all meshes on all MPI procs because
        !       the mesh file I/O has not been adapted to "each proc has its
        !       own meshes".
        ! TODO: Once the mesh I/O has been adapted, the bounds from
        !       dcomm_handler can be used. Then each proc will only build its
        !       own meshes which will speed up the initialization.
        !this%lb_phi = dcomm_handler%lb_stripped(2)
        !this%ub_phi = dcomm_handler%ub_stripped(2)
        this%lb_phi = 1
        this%ub_phi = n_points_phi

        ! If specified, read in remaining mesh information from file
        if(read_mesh_from_file_local) then
            if(present(mesh_filename)) then
                call this%read_mesh(mesh_filename, spacing_RZ, &
                                    multigrid_levels)
#ifdef ENABLE_GPU
                if(get_use_gpu_offload()) then
                    ! Initialize the mesh_5d_t C++ class
                    call this%initialize_gpu()
                endif
#endif

                ! Finish initialization of the domain decomposition handler
                call dcomm_handler%initialize_RZ_domain(&
                                                n_points_RZ=this%size_RZ_max)

                ! Construct vp_grid for BSG case and set BSG flags
                if (get_use_bsg()) then
                    call this%construct_vp_bsg()
                    call this%bsg_instance%set_flags( &
                                       this%map_positive1(this%lb_phi), &
                                       this%map_positive2(this%lb_phi), &
                                       this%map_negative1(this%lb_phi), &
                                       this%map_negative2(this%lb_phi), &
                                       this%neighbor_indices(:,:,:,this%lb_phi))
                else
                    call this%bsg_instance%construct( &
                                                this%R_buffer(:, this%lb_phi), &
                                                this%Z_buffer(:, this%lb_phi))
                end if
                return
            else
                call handle_error("Attempted to read mesh from file, but no &
                                  &filename was given!", GENEX_ERR_MESH, &
                                  __LINE__, __FILE__)
            endif
        endif

        ! Initialize multigrid from PARALLAX
        ! For the multigrid the variable reorder_size needs to be an integer
        ! array with one setting per multigrid level
        call profiler_start("initialize_multigrid", ierr, set_path=.true.)
        allocate(multigrid_reorder_size(multigrid_levels))
        multigrid_reorder_size = get_reorder_size()

        allocate(this%multigrid_3d(this%lb_phi:this%ub_phi))
        allocate(this%mesh_3d(this%lb_phi:this%ub_phi))

        if(this%is_axisymmetric()) then
            ! Initialize mesh and multigrid on first plane
            call this%multigrid_3d(this%lb_phi)%init( &
                equilibrium_instance, &
                phi=phi(this%lb_phi), &
                nlvls=multigrid_levels, &
                spacing_f=spacing_RZ, &
                size_neighbor=size_neighbor, &
                size_ghost_layer=size_ghost_layer, &
                mesh_finest=this%mesh_3d(this%lb_phi), &
                reorder_size=multigrid_reorder_size, &
                extend_beyond_wall=get_extend_beyond_wall(), &
                dbgout=get_parallax_dbgout())

            this%size_RZ_max = this%mesh_3d(this%lb_phi)%get_n_points()

            ! Copy first plane to all other planes
            do k = this%lb_phi+1, this%ub_phi
                this%mesh_3d(k) = this%mesh_3d(this%lb_phi)
                call this%mesh_3d(k)%set_phi(phi(k))

                ! Copy multigrid, correctly setting the internal mesh pointer
                call this%multigrid_3d(this%lb_phi)%copy(this%multigrid_3d(k), &
                                                    mesh_finest=this%mesh_3d(k))
                call this%multigrid_3d(k)%set_phi_coarse(phi(k))
            end do
        else
            this%size_RZ_max = 0
            do k = this%lb_phi, this%ub_phi
                call this%multigrid_3d(k)%init( &
                    equilibrium_instance, &
                    phi=phi(k), &
                    nlvls=multigrid_levels, &
                    spacing_f=spacing_RZ, &
                    size_neighbor=size_neighbor, &
                    size_ghost_layer=size_ghost_layer, &
                    mesh_finest=this%mesh_3d(k), &
                    reorder_size=multigrid_reorder_size, &
                    extend_beyond_wall=get_extend_beyond_wall(), &
                    dbgout=get_parallax_dbgout())

                this%size_RZ_max = max(this%size_RZ_max, &
                                       this%mesh_3d(k)%get_n_points())
            end do
        endif

        call profiler_stop("initialize_multigrid", ierr, set_path=.true.)

        ! The domain decomposition handler requires knowledge about the amount
        ! of points in the RZ plane to finish initialization
        call dcomm_handler%initialize_RZ_domain(n_points_RZ=this%size_RZ_max)

        ! NOTE: The order of the following initialize calls is important since
        !       some depend on the call of the previous ones
        call this%initialize_ghost_and_filler_masks()
        call this%initialize_neighbor_indices()
        call this%initialize_RZ_weights()
        call this%initialize_RZ()
        call initialize_RZ_indices(this)
        call this%initialize_jacobian()
        call this%initialize_core_mask()

        ! Construct vp_grid for BSG case
        ! Construction done here because RZ grid information is needed to obtain
        ! size of the vp domain for each block
        if (get_use_bsg()) then
            call this%construct_vp_bsg()
        else
            call this%bsg_instance%construct(this%R_buffer(:, this%lb_phi), &
                                             this%Z_buffer(:, this%lb_phi))
        end if

        call profiler_start("initialize_magfield", ierr, set_path=.true.)
        call this%initialize_magfield()
        call profiler_stop("initialize_magfield", ierr, set_path=.true.)

        call profiler_start("initialize_parcon", ierr, set_path=.true.)
        call this%initialize_parcon()
        call profiler_stop("initialize_parcon", ierr, set_path=.true.)

        call this%initialize_compute_mask()
        call this%initialize_buf_zone()

        if (get_diagnose_tpc()) then
            call this%initialize_polar()
        endif

        ! Set BSG flags
        ! Since map matrices are set in initialize_parcon() and vp_grid is
        ! required in initialize_magfield(), this call has to be done after
        ! both of them
        if(get_use_bsg()) then
            call this%bsg_instance%set_flags( &
                                       this%map_positive1(this%lb_phi), &
                                       this%map_positive2(this%lb_phi), &
                                       this%map_negative1(this%lb_phi), &
                                       this%map_negative2(this%lb_phi), &
                                       this%neighbor_indices(:,:,:,this%lb_phi))
        end if

        ! Write the mesh to file
        ! TODO_BSG: sort out I/O
        if(present(mesh_filename)) then
            call this%write_mesh(mesh_filename)
        endif

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Initialize the mesh_5d_t C++ class
            call this%initialize_gpu()
        endif
#endif

    contains

        subroutine initialize_RZ_indices(this)
            !! Initialize the mesh grid point indices in the RZ dimension
            !! NOTE: This is non-trivial in the case where the mesh is reordered
            class(mesh_5d_t), intent(inout) :: this
            integer :: n_points_RZ, n_points_RZ_inner, i, k

            n_points_RZ = this%size_RZ()
            allocate(this%RZ_indices(n_points_RZ, this%lb_phi:this%ub_phi))

            !$omp parallel default(none) &
            !$omp private(i, k, n_points_RZ_inner) &
            !$omp firstprivate(n_points_RZ) &
            !$omp shared(this)
            do k = this%lb_phi, this%ub_phi
                n_points_RZ_inner = this%mesh_3d(k)%get_n_points_inner()
            !$omp do schedule(static)
            do i = 1, n_points_RZ
                if(i <= n_points_RZ_inner) then
                    this%RZ_indices(i, k) = this%mesh_3d(k)%inner_indices(i)
                else
                    this%RZ_indices(i, k) = i
                endif
            enddo
            !$omp end do nowait
            enddo
            !$omp end parallel
        end subroutine

    end subroutine

    module subroutine initialize_ghost_and_filler_masks(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: i, k, i_bound, i_ghost, i_filler
        integer :: n_points_RZ, n_points_bound, n_points_ghost

        n_points_RZ = this%size_RZ()
        allocate(this%not_ghost(n_points_RZ, this%lb_phi:this%ub_phi))
        allocate(this%not_filler, mold=this%not_ghost)
        call this%op_set_uniform%apply(this%not_ghost, 1.0_GP)
        call this%op_set_uniform%apply(this%not_filler, 1.0_GP)

        !$omp parallel default(none) &
        !$omp private(k, n_points_bound, n_points_ghost, i_bound, i_ghost, &
        !$omp         i_filler, i) &
        !$omp firstprivate(n_points_RZ) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
            ! Combine boundary and ghost points in underlying PARALLAX mesh to
            ! total ghost points in this type
            n_points_bound = this%mesh_3d(k)%get_n_points_boundary()
            n_points_ghost = this%mesh_3d(k)%get_n_points_ghost()

            !$omp do schedule(static)
            do i_bound = 1, n_points_bound
                i = this%mesh_3d(k)%boundary_indices(i_bound)
                this%not_ghost(i, k) = 0.0_GP
            end do
            !$omp end do nowait

            !$omp do schedule(static)
            do i_ghost = 1, n_points_ghost
                i = this%mesh_3d(k)%ghost_indices(i_ghost)
                this%not_ghost(i, k) = 0.0_GP
            end do
            !$omp end do nowait

            !$omp do schedule(static)
            do i_filler = this%size_RZ_plane(k) + 1, n_points_RZ
                this%not_filler(i_filler, k) = 0.0_GP
                this%not_ghost(i_filler, k) = FILLER_VALUE_REAL
            end do
            !$omp end do nowait
        end do
        !$omp end parallel
    end subroutine

    module subroutine initialize_buf_zone(this)
        class(mesh_5d_t), target, intent(inout) :: this

        integer :: i, k, n_points_RZ
        real(kind=GP), dimension(:), pointer :: phi
        integer, dimension(:,:), pointer :: parcon_pos1, parcon_pos2, &
                                            parcon_neg1, parcon_neg2
        real(kind=GP), dimension(:), allocatable :: axis_R, axis_Z
        integer, dimension(:,:), allocatable :: buf_zone_bndry
        integer, dimension(:), allocatable :: neighbor_idxs
        integer :: size_bndry
        integer :: n_iter, i_iter, size_nbr_iter
        real(kind=GP) :: thresh_dist_to_axis, dist_to_axis, rho_center

        size_bndry = get_buf_zone_size()
        ! If a point is within this distance to the axis, it is considered an
        ! axis buffer point
        thresh_dist_to_axis = this%delta_RZ() * get_buf_zone_size_axis()

        rho_center = (this%rho_max() + this%rho_min()) / 2.0_GP

        n_points_RZ = this%size_RZ()
        phi => this%get_phi_pointer()
        parcon_pos1 => this%get_parcon_positive1_pointer()
        parcon_pos2 => this%get_parcon_positive2_pointer()
        parcon_neg1 => this%get_parcon_negative1_pointer()
        parcon_neg2 => this%get_parcon_negative2_pointer()

        allocate(this%buf_zone(n_points_RZ, this%lb_phi:this%ub_phi))
        allocate(buf_zone_bndry, mold=this%buf_zone)
        allocate(axis_R(this%lb_phi:this%ub_phi))
        allocate(axis_Z, mold=axis_R)

        call this%op_set_uniform%apply(this%buf_zone, NOT_BUF_ZONE)
        call this%op_set_uniform%apply(buf_zone_bndry, 0)

        ! Determine the axis location on every plane
        do k = this%lb_phi, this%ub_phi
            call equilibrium_instance%mag_axis_loc(phi(k), axis_R(k), axis_Z(k))
        enddo

        ! Determining the boundary buffer zone is done iteratively and stored
        ! in a temporary array since size_bndry can be larger than
        ! size_neighbor.

        ! This implementation will fail if size_neighbor is greater than
        ! size_ghost_layer, since in this case neighbors which do not
        ! exist on the mesh will be accessed.
        if(size_neighbor > size_ghost_layer) then
            call handle_error("Could not create buffer zone iteratively &
                              &because size_neighbor was larger than &
                              &size ghost!", GENEX_ERR_MESH, &
                              __LINE__, __FILE__)
        endif

        n_iter = ceiling(size_bndry * 1.0_GP / size_neighbor)

        !$omp parallel default(none) &
        !$omp private(i, k, i_iter, neighbor_idxs, size_nbr_iter, &
        !$omp         dist_to_axis) &
        !$omp firstprivate(n_points_RZ, n_iter, size_bndry, &
        !$omp              thresh_dist_to_axis, axis_R, axis_Z, rho_center) &
        !$omp shared(this, buf_zone_bndry, parcon_pos1, parcon_pos2, &
        !$omp        parcon_neg1, parcon_neg2)
        do k = this%lb_phi, this%ub_phi
        do i_iter = 1, n_iter

            ! The number of neighbors checked in this iteration is either
            ! size_neighbor or the remainder necessary to fill size_bndry
            size_nbr_iter = min(size_neighbor, &
                                size_bndry - (i_iter - 1) * size_neighbor)

            !$omp do schedule(static)
            do i = 1, n_points_RZ
                if(this%is_compute(i, k) /= 1.0_GP) cycle

                neighbor_idxs = pack(this%mesh_3d(k)%index_neighbor(&
                                             -size_nbr_iter:size_nbr_iter, &
                                             -size_nbr_iter:size_nbr_iter, &
                                             i), &
                                     mask=.true.)

                ! On first iteration, a point is within boundary buffer zone if
                ! any neighbor is a non-compute (ghost or target) point
                if(i_iter == 1) then
                    if(any(this%is_compute(neighbor_idxs, k) == 0.0_GP)) then
                        buf_zone_bndry(i, k) = i_iter
                    endif
                ! For subsequent iterations, a point is within boundary buffer
                ! zone if any neighbor was identified as boundary buffer in the
                ! previous iteration
                else
                    if(buf_zone_bndry(i, k) == 0 .and. &
                       any(buf_zone_bndry(neighbor_idxs, k) == i_iter - 1)) then
                        buf_zone_bndry(i, k) = i_iter
                    endif
                endif
            end do
            !$omp end do
        end do
        end do

        ! Determine final buffer zone
        do k = this%lb_phi, this%ub_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            if(this%is_compute(i, k) /= 1.0_GP) cycle

            ! Compute distance of current point to magnetic axis
            dist_to_axis = sqrt((this%R_buffer(i, k) - axis_R(k))**2 + &
                                (this%Z_buffer(i, k) - axis_Z(k))**2)

            ! Point is within boundary buffer zone
            if(buf_zone_bndry(i, k) > 0) then
                ! Check if point is within the inner or outer boundary
                ! buffer region.
                if(this%is_core(i, k) == 1.0_GP &
                    .and. this%rho_buffer(i, k) <= rho_center) then
                    this%buf_zone(i, k) = BUF_ZONE_BOUNDARY_IN
                else
                    this%buf_zone(i, k) = BUF_ZONE_BOUNDARY_OUT
                endif

            ! Any point which is within the specified distance from the
            ! axis is within the axis buffer zone. This requires the magnetic
            ! axis to be incluced.
            elseif(dist_to_axis < thresh_dist_to_axis) then
                this%buf_zone(i, k) = BUF_ZONE_AXIS
            ! Any point which connects to the target in the parallel
            ! direction is within the parcon buffer zone
            elseif(parcon_pos1(i, k) == PARCON_COMPUTE_TO_TARGET .or. &
                   parcon_pos2(i, k) == PARCON_COMPUTE_TO_TARGET .or. &
                   parcon_neg1(i, k) == PARCON_COMPUTE_TO_TARGET .or. &
                   parcon_neg2(i, k) == PARCON_COMPUTE_TO_TARGET) then
                this%buf_zone(i, k) = BUF_ZONE_PARCON
            endif
        end do
        !$omp end do
        end do
        !$omp end parallel

    end subroutine

    module subroutine initialize_compute_mask(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: i, k, n_points_RZ

        n_points_RZ = this%size_RZ()
        allocate(this%is_compute(n_points_RZ, this%lb_phi:this%ub_phi))

        !$omp parallel default(none) &
        !$omp private(i, k) &
        !$omp firstprivate(n_points_RZ) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
            !$omp do schedule(static)
            do i = 1, n_points_RZ
                this%is_compute(i, k) = this%not_in_target(i, k) &
                                      * this%not_ghost(i, k) &
                                      * this%not_filler(i, k)
            end do
            !$omp end do
        end do
        !$omp end parallel
    end subroutine

    module subroutine initialize_core_mask(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: i, k, n_points_RZ

        n_points_RZ = this%size_RZ()
        allocate(this%is_core(n_points_RZ, this%lb_phi:this%ub_phi))
        call this%op_set_uniform%apply(this%is_core, 0.0_GP)

        !$omp parallel default(none) &
        !$omp private(i, k) &
        !$omp firstprivate(n_points_RZ) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
            !$omp do schedule(static)
            do i = 1, n_points_RZ
                if(this%district(i, k) == DISTRICT_PRIVFLUX .or. &
                   this%district(i, k) == DISTRICT_SOL .or. &
                   this%district(i, k) == DISTRICT_WALL .or. &
                   this%district(i, k) == DISTRICT_DOME .or. &
                   this%district(i, k) == DISTRICT_OUT) then
                    this%is_core(i, k) = 0.0_GP

                elseif(this%not_filler(i, k) == 1.0_GP) then
                    this%is_core(i, k) = 1.0_GP

                else
                    this%is_core(i, k) = FILLER_VALUE_REAL
                endif
            end do
            !$omp end do
        end do
        !$omp end parallel
    end subroutine

    module subroutine initialize_neighbor_indices(this)
        class(mesh_5d_t), target, intent(inout) :: this

        integer :: n_points_RZ, i, k, nbr_R, nbr_Z
        integer, dimension(:,:,:), pointer :: neighbors_plane

        n_points_RZ = this%size_RZ()

        allocate(this%neighbor_indices(-size_neighbor:size_neighbor, &
                                       -size_neighbor:size_neighbor, &
                                       1:n_points_RZ, &
                                       this%lb_phi:this%ub_phi))

        !$omp parallel default(none) &
        !$omp firstprivate(n_points_RZ) &
        !$omp private(k, i, nbr_Z, nbr_R, neighbors_plane) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
            neighbors_plane => this%mesh_3d(k)%index_neighbor
        !$omp do
        do i = 1, n_points_RZ
            if(this%not_filler(i, k) == 1.0_GP) then
                do nbr_Z = -size_neighbor, size_neighbor
                do nbr_R = -size_neighbor, size_neighbor
                    if(neighbors_plane(nbr_R, nbr_Z, i) == 0) then
                        ! If there is no neighbor in this position, PARALLAX
                        ! returns an index of 0. Since this should only be the
                        ! case for ghost points, and will cause out-of-bounds
                        ! errors in the operators, replace with the index of
                        ! the current point
                        this%neighbor_indices(nbr_R, nbr_Z, i, k) = i
                    else
                        this%neighbor_indices(nbr_R, nbr_Z, i, k) = &
                              neighbors_plane(nbr_R, nbr_Z, i)
                    endif
                enddo
                enddo
            else
                ! For filler points, set the "neighbor" index to the current
                ! point, to avoid out-of-bounds errors in the operators
                this%neighbor_indices(:, :, i, k) = i
            endif
        enddo
        !$omp end do
        enddo
        !$omp end parallel
    end subroutine

    module subroutine construct_vp_bsg(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: nb
        real(kind=GP) :: vp_bsg_length

        call this%bsg_instance%construct(this%R_buffer(:, this%lb_phi), &
                                      this%Z_buffer(:, this%lb_phi), &
                                      is_core=this%is_core(:, this%lb_phi), &
                                      equilibrium_instance=equilibrium_instance)
        do nb = 1, get_num_bsg_blocks()
            vp_bsg_length = this%bsg_instance%get_vp_length(nb)
            call this%vp_grid(nb)%construct(vp_bsg_length, &
                                            get_quad_type_vp())
        end do
    end subroutine

    module subroutine initialize_RZ(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: i, k, n_points_RZ

        n_points_RZ  = this%size_RZ()
        allocate(this%R_buffer(n_points_RZ, this%lb_phi:this%ub_phi))
        allocate(this%Z_buffer, mold=this%R_buffer)
        !$omp parallel default(none) &
        !$omp private(i, k) &
        !$omp firstprivate(n_points_RZ) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            if(this%not_filler(i, k) == 1.0_GP) then
                this%R_buffer(i, k) = this%mesh_3d(k)%get_x(i)
                this%Z_buffer(i, k) = this%mesh_3d(k)%get_y(i)
            else
                ! Filler point
                this%R_buffer(i, k) = FILLER_VALUE_REAL
                this%Z_buffer(i, k) = FILLER_VALUE_REAL
            end if
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel
    end subroutine

    module subroutine initialize_polar(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: p_i, k, polar_n_theta, polar_n_rho

        polar_n_theta = get_polar_n_theta()
        polar_n_rho   = get_polar_n_rho()

        allocate(this%polar_theta_buffer(polar_n_theta, &
                                         this%lb_phi:this%ub_phi))
        allocate(this%polar_rho_buffer(polar_n_rho, this%lb_phi:this%ub_phi))

        !$omp parallel default(none) &
        !$omp private(p_i, k) &
        !$omp firstprivate(polar_n_theta, polar_n_rho) &
        !$omp shared(this)
        do k = this%lb_phi, this%ub_phi
        !$omp do schedule(static)
        do p_i = 1, polar_n_theta
            this%polar_theta_buffer(p_i, k) = this%mesh_3d_pol(k)%get_theta(p_i)
        enddo
        !$omp end do nowait
        enddo

        do k = this%lb_phi, this%ub_phi
        !$omp do schedule(static)
        do p_i = 1, polar_n_rho
            this%polar_rho_buffer(p_i, k) = this%mesh_3d_pol(k)%get_rho(p_i)
        enddo
        !$omp end do nowait
        enddo
        !$omp end parallel
    end subroutine

    module subroutine initialize_RZ_weights(this)
        class(mesh_5d_t), target, intent(inout) :: this

        integer :: i, k, n_points_RZ
        integer :: neigh_count

        n_points_RZ = this%size_RZ()

        allocate(this%RZw_buffer(n_points_RZ, this%lb_phi:this%ub_phi), &
                 source=0.0_GP)

        !$omp parallel default(none) &
        !$omp private(i, k, neigh_count) &
        !$omp firstprivate(n_points_RZ) &
        !$omp shared(this)
        !$omp do schedule(static)
        do k = this%lb_phi, this%ub_phi
        do i = 1, n_points_RZ
            if(this%not_filler(i, k) == 0.0_GP) cycle

            ! NOTE: Weight cannot be assumed 1 for the inner points, since these
            !       are defined such that only the 5 point stencil is guaranteed
            !       to be an inner or boundary point. In the 9 point stencil
            !       some of the neighbors can be ghost points. Thus we apply
            !       the same weight algorithm for the inner and boundary points.

            ! Weights depend on the shape on the polygon defining the boundary.
            ! This polygon is characterized by the number of the neighbors that
            ! are inner or boundary points.
            if(.not. this%mesh_3d(k)%is_ghost_point(i)) then
                neigh_count = count_neighbors(i, k)

                ! NOTE: Inner points have a different weight here, since inner
                !       points with 8 neighbors should have weight 1. Inner
                !       points with 7 neighbors are also correctly treated,
                !       because their area polygon is larger than the one of
                !       a boundary point with 7 neighbors.
                if(this%mesh_3d(k)%is_inner_point(i)) then
                    this%RZw_buffer(i, k) = neigh_count / 8.0_GP
                else
                    this%RZw_buffer(i, k) = (neigh_count - 1) / 8.0_GP
                endif
            endif
        enddo
        enddo
        !$omp end do nowait
        !$omp end parallel
        this%RZw_buffer = this%RZw_buffer * this%delta_RZ()**2

    contains

        integer function count_neighbors(i, k)
            ! Counts neighbors in 9 point stencil which are in domain
            ! (inner or boundary points)
            integer, intent(in) :: i, k

            integer :: j1, j2, counter, i_neigh

            counter = 0
            do j1 = -1, 1
            do j2 = -1, 1
                i_neigh = this%neighbor_indices(j1, j2, i, k)
                if(i == i_neigh) cycle

                if(this%mesh_3d(k)%is_inner_point(i_neigh) &
                    .or. this%mesh_3d(k)%is_boundary_point(i_neigh)) then
                    counter = counter + 1
                endif
            enddo
            enddo
            count_neighbors = counter

        end function

    end subroutine

    module subroutine initialize_jacobian(this)
        class(mesh_5d_t), target, intent(inout) :: this
        integer :: i, k, n_points_RZ, equi_type

        n_points_RZ = this%size_RZ()
        equi_type = this%equi_type()

        allocate(this%jacobian_buffer(n_points_RZ, this%lb_phi:this%ub_phi))

        ! The slab and circular equilibria are cartesian; all other equilibria
        ! are in cylindrical coordinates
        if(equi_type == SLAB .or. equi_type == CIRCULAR) then
            !$omp parallel default(none) &
            !$omp private(i, k) firstprivate(n_points_RZ) &
            !$omp shared(this)
            do k = this%lb_phi, this%ub_phi
            !$omp do schedule(static)
            do i = 1, n_points_RZ
                ! NOTE: No filler handling required for slab and circular
                this%jacobian_buffer(i, k) = 1.0_GP
            enddo
            !$omp end do nowait
            enddo
            !$omp end parallel
        else
            !$omp parallel default(none) &
            !$omp private(i, k) firstprivate(n_points_RZ) &
            !$omp shared(this)
            do k = this%lb_phi, this%ub_phi
            !$omp do schedule(static)
            do i = 1, n_points_RZ
                ! NOTE: Filler points are set by R_buffer
                this%jacobian_buffer(i, k) = this%R_buffer(i, k)
            enddo
            !$omp end do nowait
            enddo
            !$omp end parallel
        endif
    end subroutine

    module subroutine create_helmholtz_solver(this, lb, ub, &
                                              lb_stripped, ub_stripped, &
                                              co, lambda, xi, solvers)
        class(mesh_5d_t), intent(inout) :: this
        integer, dimension(2), intent(in) :: lb, ub
        integer, dimension(2), intent(in) :: lb_stripped, ub_stripped
        real(kind=GP), dimension(lb(1):ub(1), &
                                 lb_stripped(2):ub_stripped(2) &
                                 ), intent(in) :: co
        real(kind=GP), dimension(lb_stripped(1):ub_stripped(1), &
                                 lb_stripped(2):ub_stripped(2) &
                                 ), intent(in) :: lambda
        real(kind=GP), dimension(lb_stripped(1):ub_stripped(1), &
                                 lb_stripped(2):ub_stripped(2) &
                                 ), intent(in) :: xi
        class(helmholtz_solver_t), &
            dimension(lb_stripped(2):ub_stripped(2)), intent(out) :: solvers

        type(parameters_helmholtz_solver_factory) :: helm_params
        ! Parameters for the creation of the solver using the factory routine
        integer :: k
        ! Loop index
        integer :: np, np_inner
        ! Number of total and inner mesh points on each plane
        integer :: parallax_gpu_data_backend
        ! Backend type for PARALLAX explicit data management

        if(get_use_parallax_gpu_data_explicit()) then
            parallax_gpu_data_backend = PARALLAX_BACKEND_GPU
        else
            parallax_gpu_data_backend = PARALLAX_BACKEND_CPU
        endif

        ! We only use the MGMRES type solver
        helm_params%smoother_type  = get_smoother()
        helm_params%gmres_maxiter  = get_maxiter()
        ! Set nrestart to be the same as maxiter
        helm_params%gmres_nrestart = get_maxiter()
        helm_params%restol_zero    = get_restol_zero()
        ! Set the solver tolerance depending on the precision used
        if(GP == DP) then
            helm_params%rtol        = 1e-8_GP
        else if(GP == SP) then
            helm_params%rtol        = 1e-4_GP
        else
            ! Use default and give a warning
            call handle_error("Helmholtz solver precision is not defined for &
                              &compiled precision="//string(GP) &
                              //"! (default used)", &
                              GENEX_WRN_MESH, __LINE__, __FILE__)
        endif
        helm_params%dbgout = get_parallax_dbgout()

        ! Create the solver array using factory routine. Type allocation
        ! and initialization of the hsolver happens inside the factory.
        ! NOTE: MPI parallelization over RZ is not yet supported, so co, lambda
        !       and xi are indexed according to the mesh routines rather than
        !       lb and ub
        do k = lb_stripped(2), ub_stripped(2)
            np       = this%mesh_3d(k)%get_n_points()
            np_inner = this%mesh_3d(k)%get_n_points_inner()
            if(lb(1) /= 1 .or. lb_stripped(1) /= 1 .or. &
               ub(1) < np .or. ub_stripped(1) < np_inner) then
                call handle_error("Helmholtz solver coefficients have &
                                  &inconsistent bounds!", GENEX_ERR_MESH, &
                                  __LINE__, __FILE__, &
                                  additional_info=error_info_t( &
                                    "Bounds were/should (lb, lb_stripped, &
                                    &ub, ub_stripped):", &
                                    [lb(1), lb_stripped(1), ub(1), &
                                     ub_stripped(1), 1, 1, np, np_inner]))
            end if

            select type(solvers)
                type is(helmholtz_solver_mgmres_cpu_t)
                    call helmholtz_solver_factory(this%multigrid_3d(k), &
                            bnd_type_core=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_wall=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_dome=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_out =BND_TYPE_DIRICHLET_ZERO, &
                            co=co(1:np, k), &
                            lambda=lambda(1:np_inner, k), &
                            xi=xi(1:np_inner, k), &
                            par=helm_params, &
                            hsolver=solvers(k))
#ifdef ENABLE_PARALLAX_GPU
                type is(helmholtz_solver_mgmres_cxx_t)
                    call helmholtz_solver_factory(this%multigrid_3d(k), &
                            compute_backend_to_use= &
                                get_parallax_gpu_offload_backend(), &
                            bnd_type_core=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_wall=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_dome=BND_TYPE_DIRICHLET_ZERO, &
                            bnd_type_out =BND_TYPE_DIRICHLET_ZERO, &
                            co=co(1:np, k), &
                            lambda=lambda(1:np_inner, k), &
                            xi=xi(1:np_inner, k), &
                            par=helm_params, &
                            hsolver=solvers(k), &
                            data_backend_to_use=parallax_gpu_data_backend)
#endif
            end select
        end do

    end subroutine

    module subroutine trace_wrapper(this, x, y, phi, delta_phi, xp, yp, &
                                    fll, vol)
        class(mesh_5d_t), target, intent(inout) :: this
        real(kind=GP), intent(in) :: x, y, phi
        real(kind=GP), intent(out) :: xp, yp
        real(kind=GP), intent(in) :: delta_phi
        real(kind=GP), intent(out), optional :: fll, vol

        real(kind=GP) :: fll_local, vol_local

        !$omp critical
        call trace(x, y, phi, delta_phi, equilibrium_instance, &
                   x_end=xp, &
                   y_end=yp, &
                   arclen=fll_local, &
                   fluxexp=vol_local)
        !$omp end critical
        if(present(fll)) then
            fll = fll_local
        endif
        if(present(vol)) then
            vol = vol_local
        endif
    end subroutine

    module subroutine read_mesh(this, mesh_filename, spacing_RZ, &
                                multigrid_levels)
        class(mesh_5d_t), target, intent(inout) :: this
        character(len=*), intent(in) :: mesh_filename
        real(kind=GP), intent(in) :: spacing_RZ
        integer, intent(in) :: multigrid_levels

        type(mesh_file_t) :: mesh_file
        integer :: n_points_RZ, n_points_phi, polar_n_theta, polar_n_rho
        integer :: equi_type
        real(kind=GP), dimension(:), allocatable :: phi_read

        call mesh_file%initialize(this%dcomm_handler, mesh_filename, &
                                  this%size_phi(), this%is_axisymmetric(), &
                                  read_mesh=.true.)

        call mesh_file%read_dimensions(n_points_RZ, n_points_phi)

        call read_parallax(mesh_file)

        allocate(this%not_ghost(n_points_RZ, n_points_phi))
        allocate(this%not_filler, &
                 this%R_buffer, &
                 this%Z_buffer, &
                 this%RZw_buffer, &
                 this%jacobian_buffer, &
                 mold=this%not_ghost)
        allocate(this%buf_zone(n_points_RZ, n_points_phi))
        allocate(this%RZ_indices(n_points_RZ, n_points_phi))
        allocate(phi_read(n_points_phi))

        call mesh_file%read_mesh(equi_type, &
                                 this%size_RZ_max, &
                                 this%not_ghost, &
                                 this%not_filler, &
                                 this%buf_zone, &
                                 this%R_buffer, &
                                 this%Z_buffer, &
                                 this%RZw_buffer, &
                                 this%RZ_indices, &
                                 this%jacobian_buffer, &
                                 phi_read)

        allocate(this%absB_buffer, &
                 this%normb_R_buffer, &
                 this%normb_Z_buffer, &
                 this%normb_tor_buffer, &
                 this%curl_normb_y, &
                 this%dgyxdy_over_g, &
                 this%dgyzdy_over_g, &
                 this%dgyxdz_over_g, &
                 this%dgyzdx_over_g, &
                 this%inv_g, &
                 this%dabsBdx, &
                 this%dabsBdz, &
                 this%dabsBdy, &
                 this%rho_buffer, &
                 this%theta_buffer, &
                 this%psi_buffer, &
                 mold=this%not_ghost)

        call mesh_file%read_magnetic_field(this%psi_norm_fac, &
                                           this%psi_lo, &
                                           this%psi_up, &
                                           this%absB_max, &
                                           this%absB_buffer, &
                                           this%normb_R_buffer, &
                                           this%normb_Z_buffer, &
                                           this%normb_tor_buffer, &
                                           this%curl_normb_y, &
                                           this%dgyxdy_over_g, &
                                           this%dgyzdy_over_g, &
                                           this%dgyxdz_over_g, &
                                           this%dgyzdx_over_g, &
                                           this%inv_g, &
                                           this%dabsBdx, &
                                           this%dabsBdz, &
                                           this%dabsBdy, &
                                           this%rho_buffer, &
                                           this%theta_buffer, &
                                           this%psi_buffer)

        if (get_diagnose_tpc()) then
            call mesh_file%read_polar_dimensions(polar_n_theta, polar_n_rho)

            allocate(this%polar_theta_buffer(polar_n_theta, n_points_phi))
            allocate(this%polar_rho_buffer(polar_n_rho, n_points_phi))
            allocate(this%polar_absB_buffer(polar_n_theta, polar_n_rho, &
                                            n_points_phi))
            allocate(this%loss_cone(n_points_RZ, n_points_phi))
            call mesh_file%read_polar_mesh(this%polar_theta_buffer, &
                                           this%polar_rho_buffer)
            call mesh_file%read_polar_magnetic_field(this%polar_absB_buffer, &
                                                     this%loss_cone)
        endif
        allocate(this%not_in_target(n_points_RZ, n_points_phi))
        allocate(this%maps_on_mesh, &
                 this%fll_positive1, &
                 this%fll_positive2, &
                 this%fll_negative1, &
                 this%fll_negative2, &
                 mold=this%not_ghost)

        allocate(this%parcon_positive1(n_points_RZ, n_points_phi))
        allocate(this%parcon_positive2, &
                 this%parcon_negative1, &
                 this%parcon_negative2, &
                 mold=this%parcon_positive1)

        call mesh_file%read_parcon(this%not_in_target, &
                                   this%maps_on_mesh, &
                                   this%fll_positive1, &
                                   this%fll_positive2, &
                                   this%fll_negative1, &
                                   this%fll_negative2, &
                                   this%parcon_positive1, &
                                   this%parcon_positive2, &
                                   this%parcon_negative1, &
                                   this%parcon_negative2)

        call this%initialize_compute_mask()
        call this%initialize_neighbor_indices()
        call this%initialize_core_mask()

        call consistency_check()

    contains

        subroutine read_parallax(mesh_file)
            !! Read PARALLAX data types (multigrid and map matrices) from file
            !! from their own groups
            type(mesh_file_t), intent(in) :: mesh_file

            integer :: k, id
            real(kind=GP), dimension(:), pointer :: phi

            phi => this%get_phi_pointer()

            allocate(this%multigrid_3d(this%lb_phi:this%ub_phi))
            allocate(this%mesh_3d(this%lb_phi:this%ub_phi))
            allocate(this%map_positive1(this%lb_phi:this%ub_phi))
            allocate(this%map_positive2(this%lb_phi:this%ub_phi))
            allocate(this%map_negative1(this%lb_phi:this%ub_phi))
            allocate(this%map_negative2(this%lb_phi:this%ub_phi))

            ! For axisymmetric equilibria, the meshes and map matrices on
            ! all planes are the same. Only read the first plane from file
            if(this%is_axisymmetric()) then
                id = mesh_file%get_id_multigrid(1)
                call this%multigrid_3d(this%lb_phi)%read_netcdf(&
                                                id, this%mesh_3d(this%lb_phi))

                id = mesh_file%get_id_map_positive1(1)
                call this%map_positive1(this%lb_phi)%read_netcdf(id)
                id = mesh_file%get_id_map_positive2(1)
                call this%map_positive2(this%lb_phi)%read_netcdf(id)
                id = mesh_file%get_id_map_negative1(1)
                call this%map_negative1(this%lb_phi)%read_netcdf(id)
                id = mesh_file%get_id_map_negative2(1)
                call this%map_negative2(this%lb_phi)%read_netcdf(id)

                do k = this%lb_phi+1, this%ub_phi
                    ! Copy mesh from first plane, setting correct phi
                    this%mesh_3d(k) = this%mesh_3d(this%lb_phi)
                    call this%mesh_3d(k)%set_phi(phi(k))

                    ! Copy multigrid from first plane, setting correct internal
                    ! mesh pointer and phi
                    call this%multigrid_3d(this%lb_phi)%copy(&
                                                    this%multigrid_3d(k), &
                                                    this%mesh_3d(k))
                    call this%multigrid_3d(k)%set_phi_coarse(phi(k))

                    this%map_positive1(k) = this%map_positive1(this%lb_phi)
                    this%map_positive2(k) = this%map_positive2(this%lb_phi)
                    this%map_negative1(k) = this%map_negative1(this%lb_phi)
                    this%map_negative2(k) = this%map_negative2(this%lb_phi)
                enddo
            else
                do k = this%lb_phi, this%ub_phi
                    id = mesh_file%get_id_multigrid(k)
                    call this%multigrid_3d(k)%read_netcdf(id, this%mesh_3d(k))
                    id = mesh_file%get_id_map_positive1(k)
                    call this%map_positive1(k)%read_netcdf(id)
                    id = mesh_file%get_id_map_positive2(k)
                    call this%map_positive2(k)%read_netcdf(id)
                    id = mesh_file%get_id_map_negative1(k)
                    call this%map_negative1(k)%read_netcdf(id)
                    id = mesh_file%get_id_map_negative2(k)
                    call this%map_negative2(k)%read_netcdf(id)
                enddo
            endif
        end subroutine

        subroutine consistency_check()
            !! Check consistency of values read in from mesh file against
            !! provided mesh parameters

            integer :: k
            real(kind=GP), dimension(:), pointer :: phi_local

            if(this%equi_type() /= equi_type) then
                call handle_error("Equi type read from mesh file is &
                                  &inconsistent with initialized value!", &
                                  GENEX_ERR_MESH, __LINE__, __FILE__, &
                                  additional_info=error_info_t( &
                                    "Was/expected: ", &
                                    [equi_type, this%equi_type()]))
            endif
            if(this%size_RZ_max /= n_points_RZ) then
                call handle_error("Max size RZ read from mesh file is &
                                  &inconsistent with initialized value!", &
                                  GENEX_ERR_MESH, __LINE__, __FILE__, &
                                  additional_info=error_info_t( &
                                    "Was/expected: ", &
                                    [n_points_RZ, this%size_RZ_max]))
            endif
            if(this%size_phi() /= n_points_phi) then
                call handle_error("Size phi read from mesh file is &
                                  &inconsistent with initialized value!", &
                                  GENEX_ERR_MESH, __LINE__, __FILE__, &
                                  additional_info=error_info_t( &
                                    "Was/expected: ", &
                                    [n_points_phi, this%size_phi()]))
            endif

            if (allocated(this%mesh_3d_pol)) then
                do k = 1, n_points_phi

                if(this%size_polar_theta(k) /= polar_n_theta) then
                    call handle_error("Size of polar theta read from mesh file &
                                     &does not match value initialized from &
                                     &parameters!", &
                                     GENEX_ERR_MESH, __LINE__, __FILE__, &
                                     additional_info=error_info_t( &
                                     "Was/expected: ", &
                                     [polar_n_theta, this%size_polar_theta(k)]))
                endif
                if(this%size_polar_rho(k) /= polar_n_rho) then
                    call handle_error("Size of polar rho read from mesh file &
                                     &does not match value initialized from &
                                     &parameters!", &
                                     GENEX_ERR_MESH, __LINE__, __FILE__, &
                                     additional_info=error_info_t( &
                                     "Was/expected: ", &
                                     [polar_n_rho, this%size_polar_rho(k)]))
                endif

                enddo
            endif

            phi_local => this%get_phi_pointer()

            do k = 1, n_points_phi
                if(this%mesh_3d(k)%get_spacing_f() /= spacing_RZ) then
                    call handle_error("Spacing RZ read from mesh file is &
                                      &inconsistent with initialized value!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__, &
                                      additional_info=error_info_t( &
                                        "On plane, was/expected: ", [k], &
                                        [spacing_RZ, &
                                         this%mesh_3d(k)%get_spacing_f()]))
                endif

                if(this%multigrid_3d(k)%get_nlvls() /= multigrid_levels) then
                    call handle_error("Number of multigrid levels read from &
                                      &mesh file is &
                                      &inconsistent with initialized value!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__, &
                                      additional_info=error_info_t( &
                                        "On plane, was/expected: ", &
                                        [k, multigrid_levels, &
                                         this%multigrid_3d(k)%get_nlvls()]))
                endif

                if(.not. almost_equal(phi_local(k), phi_read(k), GP_EPS)) then
                    call handle_error("Phi value read from mesh file is &
                                      &inconsistent with initialized value!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__, &
                                      additional_info=error_info_t( &
                                        "On plane, was/expected: ", [k], &
                                        [phi_read(k), phi_local(k)]))
                endif

                if(.not. almost_equal(phi_local(k), &
                                      this%mesh_3d(k)%get_phi(), GP_EPS)) then
                    call handle_error("Phi value of 3D mesh read from &
                                      &mesh file is &
                                      &inconsistent with initialized value!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__, &
                                      additional_info=error_info_t( &
                                        "On plane, was/expected: ", [k], &
                                        [phi_read(k), &
                                         this%mesh_3d(k)%get_phi()]))
                endif
            enddo
        end subroutine
    end subroutine

    module subroutine write_mesh(this, mesh_filename)
        class(mesh_5d_t), target, intent(inout) :: this
        character(len=*), intent(in) :: mesh_filename

        type(mesh_file_t) :: mesh_file
        integer :: n_points_RZ, n_points_phi, i, k
        integer, dimension(:,:), allocatable :: x_ind, z_ind

        ! TODO: Fix mesh file write with multiple MPI procs in phi

        n_points_RZ  = this%size_RZ()
        n_points_phi = this%size_phi()

        call mesh_file%initialize(this%dcomm_handler, mesh_filename, &
                                  n_points_phi, this%is_axisymmetric())

        ! Create the x and z cartesian indices of the grid points
        allocate(x_ind(n_points_RZ, this%lb_phi:this%ub_phi))
        allocate(z_ind, mold=x_ind)

        do k = this%lb_phi, this%ub_phi
        do i = 1, n_points_RZ
            if(this%not_filler(i, k) == 1.0_GP) then
                x_ind(i, k) = this%mesh_3d(k)%get_cart_i(i)
                z_ind(i, k) = this%mesh_3d(k)%get_cart_j(i)
            else
                x_ind(i, k) = FILLER_VALUE_INT
                z_ind(i, k) = FILLER_VALUE_INT
            endif
        enddo
        enddo

        call write_parallax(mesh_file)

        ! TODO_BSG: Implement write_mesh for BSG
        call mesh_file%write_mesh(this%equi_type(), &
                                  this%size_RZ_max, &
                                  this%not_ghost, &
                                  this%not_filler, &
                                  this%buf_zone, &
                                  this%R_buffer, &
                                  this%Z_buffer, &
                                  this%RZw_buffer, &
                                  x_ind, &
                                  z_ind, &
                                  this%RZ_indices, &
                                  this%jacobian_buffer, &
                                  this%phi_grid%get_pointer(), &
                                  this%mu_grid%get_pointer(), &
                                  this%vp_grid(1)%get_pointer())

        call mesh_file%write_magnetic_field(this%psi_norm_fac, &
                                            this%psi_lo, &
                                            this%psi_up, &
                                            this%absB_max, &
                                            this%absB_buffer, &
                                            this%normb_R_buffer, &
                                            this%normb_Z_buffer, &
                                            this%normb_tor_buffer, &
                                            this%curl_normb_y, &
                                            this%dgyxdy_over_g, &
                                            this%dgyzdy_over_g, &
                                            this%dgyxdz_over_g, &
                                            this%dgyzdx_over_g, &
                                            this%inv_g, &
                                            this%dabsBdx, &
                                            this%dabsBdz, &
                                            this%dabsBdy, &
                                            this%rho_buffer, &
                                            this%theta_buffer, &
                                            this%psi_buffer)

        if (get_diagnose_tpc()) then
            call mesh_file%write_polar_mesh(this%polar_theta_buffer, &
                                            this%polar_rho_buffer, &
                                            this%phi_grid%get_pointer())
            call mesh_file%write_polar_magnetic_field(this%polar_absB_buffer, &
                                                      this%loss_cone)
        endif

        call mesh_file%write_parcon(this%not_in_target, &
                                    this%maps_on_mesh, &
                                    this%fll_positive1, &
                                    this%fll_positive2, &
                                    this%fll_negative1, &
                                    this%fll_negative2, &
                                    this%parcon_positive1, &
                                    this%parcon_positive2, &
                                    this%parcon_negative1, &
                                    this%parcon_negative2)
    contains

        subroutine write_parallax(mesh_file)
            !! Write PARALLAX data types (multigrid and map matrices) to file
            !! in their own groups
            type(mesh_file_t), intent(in) :: mesh_file

            integer :: k, id

            ! Only master writes the file
            if(.not. this%dcomm_handler%is_master()) return

            ! For axisymmetric equilibria, the meshes and map matrices on
            ! all planes are the same. Only write the first plane to file
            if(this%is_axisymmetric()) then
                id = mesh_file%get_id_multigrid(1)
                call this%multigrid_3d(this%lb_phi)%write_netcdf(id)
                id = mesh_file%get_id_map_positive1(1)
                call this%map_positive1(this%lb_phi)%write_netcdf(id)
                id = mesh_file%get_id_map_positive2(1)
                call this%map_positive2(this%lb_phi)%write_netcdf(id)
                id = mesh_file%get_id_map_negative1(1)
                call this%map_negative1(this%lb_phi)%write_netcdf(id)
                id = mesh_file%get_id_map_negative2(1)
                call this%map_negative2(this%lb_phi)%write_netcdf(id)
            else
                do k = this%lb_phi, this%ub_phi
                    id = mesh_file%get_id_multigrid(k)
                    call this%multigrid_3d(k)%write_netcdf(id)
                    id = mesh_file%get_id_map_positive1(k)
                    call this%map_positive1(k)%write_netcdf(id)
                    id = mesh_file%get_id_map_positive2(k)
                    call this%map_positive2(k)%write_netcdf(id)
                    id = mesh_file%get_id_map_negative1(k)
                    call this%map_negative1(k)%write_netcdf(id)
                    id = mesh_file%get_id_map_negative2(k)
                    call this%map_negative2(k)%write_netcdf(id)
                enddo
            endif
        end subroutine

    end subroutine

    module subroutine finalize(this)
        type(mesh_5d_t), intent(inout) :: this

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Finalize the mesh_5d_t C++ class
            call this%finalize_gpu()
        endif
#endif

        deallocate(this%multigrid_3d)
        deallocate(this%mesh_3d)
        if (allocated(this%mesh_3d_pol)) then
            deallocate(this%mesh_3d_pol)
        endif
        deallocate(equilibrium_instance)
        is_equilibrium_initialized = .false.

        if(.not. get_swap_mesh_members()) then
            ! Deallocate Fortran pointers of members related to the mesh
            deallocate(this%neighbor_indices)
            deallocate(this%buf_zone)
            deallocate(this%not_filler)
            deallocate(this%is_compute)
            deallocate(this%jacobian_buffer)
            deallocate(this%RZ_indices)

            ! Deallocate Fortran pointers of members related to the
            ! magnetic field
            deallocate(this%absB_buffer)
            if (allocated(this%polar_absB_buffer)) then
                deallocate(this%polar_absB_buffer)
            endif
            if (allocated(this%loss_cone)) then
                deallocate(this%loss_cone)
            endif
            deallocate(this%normb_R_buffer)
            deallocate(this%normb_Z_buffer)
            deallocate(this%curl_normb_y)
            deallocate(this%dgyxdy_over_g)
            deallocate(this%dgyzdy_over_g)
            deallocate(this%dgyxdz_over_g)
            deallocate(this%dgyzdx_over_g)
            deallocate(this%inv_g)
            deallocate(this%dabsBdx)
            deallocate(this%dabsBdz)
            deallocate(this%dabsBdy)

            ! Deallocate Fortran pointers of members related to the parallel
            ! connection
            deallocate(this%fll_positive1)
            deallocate(this%fll_positive2)
            deallocate(this%fll_negative1)
            deallocate(this%fll_negative2)
            deallocate(this%not_in_target)
        endif
    end subroutine

end submodule
