submodule (mesh_5d_m) mesh_5d_parcon_s
    !! This module contains the implementation of the parallel connection.
    !! This includes information of points along a magnetic field line.
    !! It can be used to calculate derivatives and boundary conditions
    !! along magnetic field lines.
    use params_mesh_m, only: get_only_first_field_period
    use params_numerical_scheme_m, only: get_int_order
    use params_devtools_m, only: get_parallax_dbgout
    use profiler_m, only: profiler_start, profiler_stop
    ! From PARALLAX
    use descriptors_m, only: DISTRICT_SOL, DISTRICT_WALL, DISTRICT_PRIVFLUX
    use map_factory_m, only: create_map_matrix, calc_dphi
    implicit none

contains

    logical function condition_in_target(x, z, phi) result(in_target)
        !! Returns .true. if the given point is in the target (divertor,
        !! limiter, or wall) and .false. otherwise
        real(kind=GP), intent(in) :: x, z, phi
        !! Coordinates of the point
        integer :: district

        district = equilibrium_instance%district(x, z, phi)

        in_target = .not. equilibrium_instance%in_vessel(x, z, phi)

        if(.not. equilibrium_instance%is_axisymmetric()) then
            in_target = in_target .and. (district == DISTRICT_WALL)
        else
            in_target = in_target .and. (district == DISTRICT_SOL .or. &
                                         district == DISTRICT_PRIVFLUX)
        endif
    end function

    subroutine create_map_matrix_wrapper(this, base_k, step, matrix, fll_array)
        !! Creates a map matrix from the base mesh at phi index k to the target
        !! mesh at phi index k + step. The map matrix gives which target mesh
        !! grid points every base mesh grid point is connected to via field
        !! lines. Additionally, returns the field line lengths from tracing each
        !! base mesh point to the target mesh.
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the mesh
        integer, intent(in) :: base_k
        !! Phi index of base mesh of the map
        integer, intent(in) :: step
        !! Relative index step to the target of the map (may be positive or
        !! negative). Map will start at phi(base_k) and cover a toroidal
        !! distance of step * delta_phi.
        type(csrmat_t), intent(out) :: matrix
        !! Created map matrix
        real(kind=GP), dimension(:), intent(out) :: fll_array
        !! Length of field lines when tracing every base mesh point to the
        !! target mesh

        integer :: n_points_RZ, n_points_phi, size_RZ_plane, i
        integer :: target_k
        integer :: n_turns
        integer, dimension(:), allocatable :: i_temp
        real(kind=GP), pointer, dimension(:) :: phi
        real(kind=GP) :: fll

        n_points_RZ  = this%size_RZ()
        n_points_phi = this%size_phi()
        phi => this%get_phi_pointer()

        ! NOTE: Modulo will wrap around size phi such that 0 is inclusive and
        !       size_phi is exclusive. Therefore the minus 1 inside and
        !       plus 1 outside is needed.
        target_k = modulo(base_k + step - 1, n_points_phi) + 1

        ! If the number of planes is less than or equal to the width of the
        ! parallel stencil, the dphi calculation may need to include extra turns
        if(n_points_phi == 1) then
            n_turns = step
        elseif(n_points_phi == 2) then
            n_turns = step / 2
        else
            n_turns = 0
        endif

        select type(equilibrium_instance)
        type is(slab_t)
            ! NOTE: For the slab equilibrium, the map matrices are equal to the
            !       identity matrix and the field line lengths are equal to the
            !       absolute difference in toroidal angle between the target and
            !       base planes. Not calling the create map matrix routine
            !       improves the performance of the routine significantly.
            matrix%nnz = n_points_RZ
            matrix%ndim = n_points_RZ
            matrix%ncol = n_points_RZ
            allocate(matrix%val(matrix%nnz))
            allocate(matrix%i(matrix%ndim + 1))
            allocate(matrix%j(matrix%nnz))

            ! Calculate dphi in the range [0, +/-2pi), with the sign matching
            ! that of the step. The field line length is the magnitude of this.
            fll = abs(calc_dphi(phi(base_k), phi(target_k), step, &
                                n_turns=n_turns))

            !$omp parallel do schedule(static) default(none) &
            !$omp private(i) firstprivate(n_points_RZ, fll) &
            !$omp shared(matrix, fll_array)
            do i = 1, n_points_RZ
                matrix%val(i) = 1.0_GP
                matrix%j(i) = i
                matrix%i(i) = i

                fll_array(i) = fll
            enddo
            !$omp end parallel do
            matrix%i(n_points_RZ + 1) = n_points_RZ + 1

        class default
            call create_map_matrix(map_matrix=matrix, &
                                   comm=this%dcomm_handler%get_comm_cart(), &
                                   equi=equilibrium_instance, &
                                   mesh_base=this%mesh_3d(base_k), &
                                   mesh_target=this%mesh_3d(target_k), &
                                   phi_direction=step, &
                                   only_first_field_period=&
                                                get_only_first_field_period(), &
                                   n_turns=n_turns, &
                                   intorder=get_int_order(), &
                                   xorder=0, &
                                   arclength_array=fll_array, &
                                   dbgout=get_parallax_dbgout(), &
                                   only_inner_points=.true.)

            ! For 3D meshes, the map matrix needs to be adjusted to include
            ! empty rows for filler points, and the field line lengths need the
            ! filler value included
            size_RZ_plane = this%size_RZ_plane(base_k)
            if(size_RZ_plane < n_points_RZ) then
                ! Copy previous array
                allocate(i_temp(size_RZ_plane + 1))
                do i = 1, size_RZ_plane + 1
                    i_temp(i) = matrix%i(i)
                enddo
                ! Reallocate matrix with space for fillers
                deallocate(matrix%i)
                allocate(matrix%i(n_points_RZ + 1))
                do i = 1, size_RZ_plane + 1
                    matrix%i(i) = i_temp(i)
                enddo
                ! Add empty rows for filler points
                do i = size_RZ_plane + 2, n_points_RZ + 1
                    matrix%i(i) = matrix%nnz + 1
                enddo
                matrix%ndim = n_points_RZ

                ! Insert filler value into fll array
                fll_array(size_RZ_plane+1:) = FILLER_VALUE_REAL
            endif

        end select

    end subroutine

    module subroutine initialize_parcon(this)
        class(mesh_5d_t), intent(inout) :: this
        integer :: ierr

        call profiler_start("initialize_map_matrices", ierr, set_path=.true.)
        call initialize_map_matrices(this)
        call profiler_stop("initialize_map_matrices", ierr, set_path=.true.)

        call profiler_start("initialize_parcon_masks", ierr, set_path=.true.)
        call initialize_parcon_masks(this)
        call profiler_stop("initialize_parcon_masks", ierr, set_path=.true.)

        call profiler_start("initialize_parcon_arrays", ierr, set_path=.true.)
        call initialize_parcon_arrays(this)
        call profiler_stop("initialize_parcon_arrays", ierr, set_path=.true.)

    end subroutine

    subroutine initialize_map_matrices(this)
        !! Initializes the map matrices
        class(mesh_5d_t), intent(inout) :: this
        !! Instance of the mesh

        integer :: k, n_points_RZ, n_points_phi, equi_type

        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()

        allocate(this%map_positive1(n_points_phi))
        allocate(this%map_positive2(n_points_phi))
        allocate(this%map_negative1(n_points_phi))
        allocate(this%map_negative2(n_points_phi))

        allocate(this%fll_positive1(n_points_RZ, n_points_phi))
        allocate(this%fll_positive2, &
                 this%fll_negative1, &
                 this%fll_negative2, &
                 mold=this%fll_positive1)

        ! Create maps from each plane to its neighbors, as well as the field
        ! line length arrays
        ! NOTE: In the axisymmetric case we only need to create the map matrices
        !       and fll arrays for the first plane and copy them to the other
        !       planes. This speeds up the initialization significantly. Note
        !       also that there is no filler handling necessary in this case.
        if(this%is_axisymmetric()) then
            k = 1

            call create_map_matrix_wrapper(this, k,  1, &
                                           this%map_positive1(k), &
                                           this%fll_positive1(:, k))
            call create_map_matrix_wrapper(this, k,  2, &
                                           this%map_positive2(k), &
                                           this%fll_positive2(:, k))
            call create_map_matrix_wrapper(this, k, -1, &
                                           this%map_negative1(k), &
                                           this%fll_negative1(:, k))
            call create_map_matrix_wrapper(this, k, -2, &
                                           this%map_negative2(k), &
                                           this%fll_negative2(:, k))
            do k = 2, n_points_phi
                this%map_positive1(k) = this%map_positive1(1)
                this%map_positive2(k) = this%map_positive2(1)
                this%map_negative1(k) = this%map_negative1(1)
                this%map_negative2(k) = this%map_negative2(1)

                this%fll_positive1(:, k) = this%fll_positive1(:, 1)
                this%fll_positive2(:, k) = this%fll_positive2(:, 1)
                this%fll_negative1(:, k) = this%fll_negative1(:, 1)
                this%fll_negative2(:, k) = this%fll_negative2(:, 1)
            end do
        else
            do k = 1, n_points_phi
                call create_map_matrix_wrapper(this, k,  1, &
                                               this%map_positive1(k), &
                                               this%fll_positive1(:, k))
                call create_map_matrix_wrapper(this, k,  2, &
                                               this%map_positive2(k), &
                                               this%fll_positive2(:, k))
                call create_map_matrix_wrapper(this, k, -1, &
                                               this%map_negative1(k), &
                                               this%fll_negative1(:, k))
                call create_map_matrix_wrapper(this, k, -2, &
                                               this%map_negative2(k), &
                                               this%fll_negative2(:, k))
            end do
        end if
    end subroutine

    subroutine initialize_parcon_masks(this)
        !! Initializes the not_in_target and maps_on_mesh masks
        class(mesh_5d_t), intent(inout) :: this
        !! Instance of the mesh

        integer :: i, k
        integer :: n_points_RZ, n_points_phi, equi_type, district
        real(kind=GP) :: x, z, phi
        logical :: maps_off_mesh_pos1, maps_off_mesh_pos2, &
                   maps_off_mesh_neg1, maps_off_mesh_neg2

        equi_type = this%equi_type()
        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()

        allocate(this%not_in_target(n_points_RZ, n_points_phi))
        allocate(this%maps_on_mesh(n_points_RZ, n_points_phi))

        do k = 1, n_points_phi
        do i = 1, n_points_RZ
            if(equi_type == SLAB) then
                this%maps_on_mesh(i, k) = 1.0_GP
                this%not_in_target(i, k) = 1
                ! The slab equilibrium can either represent a scrape-off
                ! layer slice with a divertor target or a periodic closed
                ! field line region. A point can only be in the target
                ! when the slab equilibrium represent a scrape-off layer
                ! slice and thus the point is part of the scrape-off layer.
                if(this%district(i, k) == DISTRICT_SOL .or. &
                   this%district(i, k) == DISTRICT_WALL) then
                    ! The first plane and the last plane represent the
                    ! target plates
                    if(k == 1 .or. k == n_points_phi) then
                        this%not_in_target(i, k) = 0
                    endif
                endif
            else
                if(this%not_filler(i, k) == 1.0_GP) then
                    ! Determine whether the current point maps off the
                    ! neighboring meshes using the map matrices
                    maps_off_mesh_pos1 = this%map_positive1(k)%i(i) &
                                       > this%map_positive1(k)%i(i + 1) - 1
                    maps_off_mesh_pos2 = this%map_positive2(k)%i(i) &
                                       > this%map_positive2(k)%i(i + 1) - 1
                    maps_off_mesh_neg1 = this%map_negative1(k)%i(i) &
                                       > this%map_negative1(k)%i(i + 1) - 1
                    maps_off_mesh_neg2 = this%map_negative2(k)%i(i) &
                                       > this%map_negative2(k)%i(i + 1) - 1
                    if(maps_off_mesh_pos1 .or. maps_off_mesh_pos2 .or. &
                       maps_off_mesh_neg1 .or. maps_off_mesh_neg2) then
                        this%maps_on_mesh(i, k) = 0.0_GP
                    else
                        this%maps_on_mesh(i, k) = 1.0_GP
                    endif

                    call this%RZphi(i, k, x, z, phi)
                    if(condition_in_target(x, z, phi)) then
                        this%not_in_target(i, k) = 0
                    else
                        this%not_in_target(i, k) = 1
                    endif
                else
                    ! Filler point
                    this%not_in_target(i, k) = FILLER_VALUE_INT
                    this%maps_on_mesh(i, k)  = FILLER_VALUE_REAL
                endif
            endif
        enddo
        enddo
    end subroutine

    subroutine initialize_parcon_arrays(this)
        !! Initializes the arrays which indicate parallel connection to the
        !! target
        class(mesh_5d_t), intent(inout) :: this
        !! Instance of the mesh

        type(csrmat_t), dimension(:), pointer :: map_p, map_pp, map_m, map_mm
        integer, dimension(:), allocatable :: map_idxs
        integer :: k_pos1, k_pos2, k_neg1, k_neg2, &
                   i, k, n_points_RZ, n_points_phi, equi_type, district

        equi_type = this%equi_type()
        n_points_RZ = this%size_RZ()
        n_points_phi = this%size_phi()
        map_p  => this%get_map_positive1_pointer()
        map_pp => this%get_map_positive2_pointer()
        map_m  => this%get_map_negative1_pointer()
        map_mm => this%get_map_negative2_pointer()

        allocate(this%parcon_positive1(n_points_RZ, n_points_phi), &
                 this%parcon_positive2(n_points_RZ, n_points_phi), &
                 this%parcon_negative1(n_points_RZ, n_points_phi), &
                 this%parcon_negative2(n_points_RZ, n_points_phi))

        !$omp parallel default(none) &
        !$omp firstprivate(n_points_RZ, n_points_phi, equi_type) &
        !$omp private(i, k, district, k_pos1, k_pos2, k_neg1, k_neg2, &
        !$omp         map_idxs) &
        !$omp shared(this, map_p, map_pp, map_m, map_mm)
        do k = 1, n_points_phi
            ! Calculate the phi indices of the neighbors to the current plane k
            k_pos1 = modulo(k + 1 - 1, n_points_phi) + 1
            k_pos2 = modulo(k + 2 - 1, n_points_phi) + 1
            k_neg1 = modulo(k - 1 - 1, n_points_phi) + 1
            k_neg2 = modulo(k - 2 - 1, n_points_phi) + 1
        !$omp do schedule(static)
        do i = 1, n_points_RZ
            ! Set filler points to the filler value
            if(this%not_filler(i, k) == 0.0_GP) then
                this%parcon_positive1(i, k) = FILLER_VALUE_INT
                this%parcon_positive2(i, k) = FILLER_VALUE_INT
                this%parcon_negative1(i, k) = FILLER_VALUE_INT
                this%parcon_negative2(i, k) = FILLER_VALUE_INT
                cycle
            endif

            if(equi_type /= SLAB) then
                ! Determine whether the current point connects target region to
                ! compute region in the parallel direction using the map
                ! matrices

                ! Plane k + 1
                ! Indices of interpolation map
                map_idxs = map_p(k)%j(map_p(k)%i(i):map_p(k)%i(i + 1) - 1)
                ! Point outside the target connects to a point inside the target
                if(this%not_in_target(i, k) == 1 .and. &
                   any(this%not_in_target(map_idxs, k_pos1) == 0)) then
                    this%parcon_positive1(i, k) = PARCON_COMPUTE_TO_TARGET

                ! Point inside the target connects to a point outside the target
                elseif(this%not_in_target(i, k) == 0 .and. &
                       any(this%not_in_target(map_idxs, k_pos1) == 1)) then
                    this%parcon_positive1(i, k) = PARCON_TARGET_TO_COMPUTE

                ! Otherwise, there is no parallel connection
                else
                    this%parcon_positive1(i, k) = NO_PARCON
                endif

                ! Plane k + 2
                map_idxs = map_pp(k)%j(map_pp(k)%i(i):map_pp(k)%i(i + 1) - 1)
                if(this%not_in_target(i, k) == 1 .and. &
                   any(this%not_in_target(map_idxs, k_pos2) == 0)) then
                    this%parcon_positive2(i, k) = PARCON_COMPUTE_TO_TARGET
                elseif(this%not_in_target(i, k) == 0 .and. &
                       any(this%not_in_target(map_idxs, k_pos2) == 1)) then
                    this%parcon_positive2(i, k) = PARCON_TARGET_TO_COMPUTE
                else
                    this%parcon_positive2(i, k) = NO_PARCON
                endif

                ! Plane k - 1
                map_idxs = map_m(k)%j(map_m(k)%i(i):map_m(k)%i(i + 1) - 1)
                if(this%not_in_target(i, k) == 1 .and. &
                   any(this%not_in_target(map_idxs, k_neg1) == 0)) then
                    this%parcon_negative1(i, k) = PARCON_COMPUTE_TO_TARGET
                elseif(this%not_in_target(i, k) == 0 .and. &
                       any(this%not_in_target(map_idxs, k_neg1) == 1)) then
                    this%parcon_negative1(i, k) = PARCON_TARGET_TO_COMPUTE
                else
                    this%parcon_negative1(i, k) = NO_PARCON
                endif

                ! Plane k - 2
                map_idxs = map_mm(k)%j(map_mm(k)%i(i):map_mm(k)%i(i + 1) - 1)
                if(this%not_in_target(i, k) == 1 .and. &
                   any(this%not_in_target(map_idxs, k_neg2) == 0)) then
                    this%parcon_negative2(i, k) = PARCON_COMPUTE_TO_TARGET
                elseif(this%not_in_target(i, k) == 0 .and. &
                       any(this%not_in_target(map_idxs, k_neg2) == 1)) then
                    this%parcon_negative2(i, k) = PARCON_TARGET_TO_COMPUTE
                else
                    this%parcon_negative2(i, k) = NO_PARCON
                endif
            else
                district = this%district(i, k)

                this%parcon_positive1(i, k) = NO_PARCON
                this%parcon_positive2(i, k) = NO_PARCON
                this%parcon_negative1(i, k) = NO_PARCON
                this%parcon_negative2(i, k) = NO_PARCON
                ! The slab equilibrium can either represent a scrape-off
                ! layer slice with a divertor target or a periodic closed
                ! field line region. A parallel connection to the target
                ! is only present in the scrape-off layer.
                if(district == DISTRICT_SOL) then
                    if(k == 1) then
                        this%parcon_positive1(i, k) = PARCON_TARGET_TO_COMPUTE
                        this%parcon_positive2(i, k) = PARCON_TARGET_TO_COMPUTE
                    elseif(k == 2) then
                        this%parcon_negative1(i, k) = PARCON_COMPUTE_TO_TARGET
                        this%parcon_negative2(i, k) = PARCON_COMPUTE_TO_TARGET
                    elseif(k == 3) then
                        this%parcon_negative2(i, k) = PARCON_COMPUTE_TO_TARGET
                    elseif(k == n_points_phi - 2) then
                        this%parcon_positive2(i, k) = PARCON_COMPUTE_TO_TARGET
                    elseif(k == n_points_phi - 1) then
                        this%parcon_positive1(i, k) = PARCON_COMPUTE_TO_TARGET
                        this%parcon_positive2(i, k) = PARCON_COMPUTE_TO_TARGET
                    elseif(k == n_points_phi) then
                        this%parcon_negative1(i, k) = PARCON_TARGET_TO_COMPUTE
                        this%parcon_negative2(i, k) = PARCON_TARGET_TO_COMPUTE
                    endif
                endif
            endif
        enddo
        !$omp end do
        enddo
        !$omp end parallel

    end subroutine

end submodule
