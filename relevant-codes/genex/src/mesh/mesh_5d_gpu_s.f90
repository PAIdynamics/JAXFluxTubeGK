submodule(mesh_5d_m) mesh_5d_gpu_s
    use, intrinsic :: iso_c_binding, only: C_CHAR, c_loc, c_f_pointer
    use genex_fortran_env_m, only: GP
    use genex_error_handling_m, only: handle_error, handle_error_gpu, &
                                      GPU_ERR_INITIALIZE, GPU_ERR_FINALIZE, &
                                      error_info_t
    use type_converters_m, only: string, string_f2c
    use mesh_5d_gpu_m, only: mesh_5d_data_t, cbind_mesh_5d_initialize, &
                             cbind_mesh_5d_finalize
    use params_gpu_offload_m, only: get_swap_mesh_members

    implicit none

contains

    module subroutine initialize_gpu(this)
        class(mesh_5d_t), intent(inout) :: this

        type(mesh_5d_data_t), allocatable :: mesh_5d_data
        ! Fortran/C++ interoperable structure for mesh_5d_t class members
        type(csrmat_data_t), dimension(:), allocatable :: map_positive1_data
        ! Fortran/C++ interoperable structure type of the map_positive1
        type(csrmat_data_t), dimension(:), allocatable :: map_negative1_data
        ! Fortran/C++ interoperable structure type of the map negative1
        type(csrmat_data_t), dimension(:), allocatable :: map_positive2_data
        ! Fortran/C++ interoperable structure type of the map_positive2
        type(csrmat_data_t), dimension(:), allocatable :: map_negative2_data
        ! Fortran/C++ interoperable structure type of the map negative2
        character(len=1, kind=C_CHAR), dimension(24) :: grid_type_phi_c
        ! C characters specifying the type of phi grid
        character(len=1, kind=C_CHAR), dimension(24) :: grid_type_vp_c
        ! C characters specifying the type of vp grid
        character(len=1, kind=C_CHAR), dimension(24) :: grid_type_mu_c
        ! C characters specifying the type of mu grid
        character(len=1, kind=C_CHAR), dimension(24) :: quad_type_phi_c
        ! C characters specifying the type of phi quadrature
        character(len=1, kind=C_CHAR), dimension(24) :: quad_type_vp_c
        ! C characters specifying the type of vp quadrature
        character(len=1, kind=C_CHAR), dimension(24) :: quad_type_mu_c
        ! C characters specifying the type of mu quadrature
        integer :: k, ierr
        ! phi loop index, C++ error status

        ! Allocate arrays of interoperable structure of the map matices
        allocate(map_positive1_data(this%lb_phi:this%ub_phi))
        allocate(map_negative1_data, mold=map_positive1_data)
        allocate(map_positive2_data, mold=map_positive1_data)
        allocate(map_negative2_data, mold=map_positive1_data)

        ! Expose class members of the map matrices
        ! TODO: Change to expose_data if PARALLAX version gets updated
        do k = this%lb_phi, this%ub_phi
            call this%map_positive1(k)%expose_data(map_positive1_data(k))
            call this%map_negative1(k)%expose_data(map_negative1_data(k))
            call this%map_positive2(k)%expose_data(map_positive2_data(k))
            call this%map_negative2(k)%expose_data(map_negative2_data(k))
        enddo

        ! Copy and convert the string members related to the mesh to C char
        call string_f2c(this%grid_type_phi(), grid_type_phi_c)
        call string_f2c(this%grid_type_vp(), grid_type_vp_c)
        call string_f2c(this%grid_type_mu(), grid_type_mu_c)
        call string_f2c(this%quad_type_phi(), quad_type_phi_c)
        call string_f2c(this%quad_type_vp(), quad_type_vp_c)
        call string_f2c(this%quad_type_mu(), quad_type_mu_c)

        ! Expose class members to the coressponding interoperable structure
        allocate(mesh_5d_data)
        call expose_data(this, map_positive1_data, map_negative1_data, &
                         map_positive2_data, map_negative2_data, &
                         mesh_5d_data)

        ! Initialize mesh_5d_t C++ class, including deep copy to the device
        ierr = cbind_mesh_5d_initialize(mesh_5d_data, &
                                        grid_type_phi_c, grid_type_vp_c, &
                                        grid_type_mu_c, quad_type_phi_c, &
                                        quad_type_vp_c, quad_type_mu_c, &
                                        this%mesh_5d_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

        if(get_swap_mesh_members()) then
            call swap_pointers(this, mesh_5d_data)
        endif

        deallocate(map_positive1_data)
        deallocate(map_negative1_data)
        deallocate(map_positive2_data)
        deallocate(map_negative2_data)
        deallocate(mesh_5d_data)

    contains

        subroutine expose_data(this, map_p_data, map_m_data, map_pp_data, &
                               map_mm_data, mesh_5d_data)
            !! Expose class members of mesh_5d_t to a Fortran/C++ interoperable
            !! structure based on mesh_5d_data_t
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
            type(csrmat_data_t), target, dimension(:), intent(in) :: map_p_data
            !! Fortran/C++ interoperable structure of map_positive1
            type(csrmat_data_t), target, dimension(:), intent(in) :: map_m_data
            !! Fortran/C++ interoperable structure of map_negative1
            type(csrmat_data_t), target, dimension(:), intent(in) :: map_pp_data
            !! Fortran/C++ interoperable structure of map_positive2
            type(csrmat_data_t), target, dimension(:), intent(in) :: map_mm_data
            !! Fortran/C++ interoperable structure of map_negative2
            type(mesh_5d_data_t), intent(inout) :: mesh_5d_data
            !! Fortran/C++ interoperable structure of mesh_5d_t

            ! Necessary to avoid Intel issue with temporary array
            real(kind=GP), contiguous, pointer, dimension(:) :: vp_grid_ptr
            real(kind=GP), contiguous, pointer, dimension(:) :: mu_grid_ptr
            real(kind=GP), contiguous, pointer, dimension(:) :: sqrt_mu_grid_ptr
            real(kind=GP), contiguous, pointer, dimension(:) :: vp_weights_ptr
            real(kind=GP), contiguous, pointer, dimension(:) :: mu_weights_ptr

            vp_grid_ptr         => this%vp_grid(1)%get_pointer()
            mu_grid_ptr         => this%mu_grid%get_pointer()
            sqrt_mu_grid_ptr    => this%mu_grid%get_sqrt_pointer()
            vp_weights_ptr      => this%vp_grid(1)%get_weights_pointer()
            mu_weights_ptr      => this%mu_grid%get_weights_pointer()

            ! Copy the value of the scalar members
            mesh_5d_data%size_RZ          = this%size_RZ()
            mesh_5d_data%size_phi         = this%size_phi()
            mesh_5d_data%size_vp          = this%size_vp()
            mesh_5d_data%size_mu          = this%size_mu()
            mesh_5d_data%size_sp          = this%size_sp()
            mesh_5d_data%order_RZ_stencil = size_neighbor
            mesh_5d_data%lb_phi           = this%lb_phi
            mesh_5d_data%ub_phi           = this%ub_phi
            mesh_5d_data%delta_RZ         = this%delta_RZ()
            mesh_5d_data%delta_phi        = this%delta_phi()
            mesh_5d_data%delta_vp         = this%delta_vp()
            mesh_5d_data%delta_sqrt_mu    = this%delta_sqrt_mu()

            ! Assign the C pointer to the array members related to the mesh
            mesh_5d_data%neighbors_ptr       = c_loc(this%neighbor_indices)
            mesh_5d_data%buf_zone_ptr        = c_loc(this%buf_zone)
            mesh_5d_data%RZ_indices_ptr      = c_loc(this%RZ_indices)
            mesh_5d_data%not_filler_ptr      = c_loc(this%not_filler)
            mesh_5d_data%is_compute_ptr      = c_loc(this%is_compute)
            mesh_5d_data%vp_grid_ptr         = c_loc(vp_grid_ptr)
            mesh_5d_data%mu_grid_ptr         = c_loc(mu_grid_ptr)
            mesh_5d_data%sqrt_mu_grid_ptr    = c_loc(sqrt_mu_grid_ptr)
            mesh_5d_data%vp_weights_ptr      = c_loc(vp_weights_ptr)
            mesh_5d_data%mu_weights_ptr      = c_loc(mu_weights_ptr)
            mesh_5d_data%jacobian_buffer_ptr = c_loc(this%jacobian_buffer)

            ! Assign the C pointer to the array members related to the
            ! magnetic field
            mesh_5d_data%absB_buffer_ptr    = c_loc(this%absB_buffer)
            mesh_5d_data%normb_R_buffer_ptr = c_loc(this%normb_R_buffer)
            mesh_5d_data%normb_Z_buffer_ptr = c_loc(this%normb_Z_buffer)
            mesh_5d_data%curl_normb_y_ptr   = c_loc(this%curl_normb_y)
            mesh_5d_data%dgyxdy_over_g_ptr  = c_loc(this%dgyxdy_over_g)
            mesh_5d_data%dgyzdy_over_g_ptr  = c_loc(this%dgyzdy_over_g)
            mesh_5d_data%dgyxdz_over_g_ptr  = c_loc(this%dgyxdz_over_g)
            mesh_5d_data%dgyzdx_over_g_ptr  = c_loc(this%dgyzdx_over_g)
            mesh_5d_data%inv_g_ptr          = c_loc(this%inv_g)
            mesh_5d_data%dabsBdx_ptr        = c_loc(this%dabsBdx)
            mesh_5d_data%dabsBdz_ptr        = c_loc(this%dabsBdz)
            mesh_5d_data%dabsBdy_ptr        = c_loc(this%dabsBdy)

            ! Assign the C pointer to the array members related to parallel
            ! connection
            mesh_5d_data%map_positive1_data_ptr = c_loc(map_p_data)
            mesh_5d_data%map_negative1_data_ptr = c_loc(map_m_data)
            mesh_5d_data%map_positive2_data_ptr = c_loc(map_pp_data)
            mesh_5d_data%map_negative2_data_ptr = c_loc(map_mm_data)
            mesh_5d_data%fll_positive1_ptr      = c_loc(this%fll_positive1)
            mesh_5d_data%fll_positive2_ptr      = c_loc(this%fll_positive2)
            mesh_5d_data%fll_negative1_ptr      = c_loc(this%fll_negative1)
            mesh_5d_data%fll_negative2_ptr      = c_loc(this%fll_negative2)
            mesh_5d_data%not_in_target_ptr      = c_loc(this%not_in_target)
        end subroutine

        subroutine swap_pointers(this, mesh_5d_data)
            !! Swap Fortran pointer of interoperable mesh members
            !! with C++ pointer
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
            type(mesh_5d_data_t), intent(inout) :: mesh_5d_data
            !! Fortran/C++ interoperable structure of mesh_5d_t

            integer :: n_RZ, n_phi
            integer :: shp_3d(2), shp_neigh(4)

            n_RZ         = this%size_RZ()
            n_phi        = this%ub_phi - this%lb_phi + 1
            shp_3d(:)    = [n_RZ, n_phi]
            shp_neigh(:) = [2 * size_neighbor + 1, 2 * size_neighbor + 1, &
                            n_RZ, n_phi]

            ! NOTE: The mesh object does not have ownership over vp_grid,
            !       mu_grid, sqrt_mu_grid, vp_weights, mu_weights,
            !       and the map matrices.
            !       Hence their pointers are not swapped.
            ! TODO: Swap the pointers of these members

            ! Deallocate Fortran pointers of members related to the mesh
            deallocate(this%neighbor_indices)
            deallocate(this%buf_zone)
            deallocate(this%RZ_indices)
            deallocate(this%not_filler)
            deallocate(this%is_compute)
            deallocate(this%jacobian_buffer)

            ! Deallocate Fortran pointers of members related to the
            ! magnetic field
            deallocate(this%absB_buffer)
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

            ! Swap the Fortran/C++ pointers of members related to the mesh
            call c_f_pointer(mesh_5d_data%neighbors_ptr, &
                             this%neighbor_indices, shp_neigh)
            call c_f_pointer(mesh_5d_data%buf_zone_ptr, this%buf_zone, shp_3d)
            call c_f_pointer(mesh_5d_data%RZ_indices_ptr, &
                             this%RZ_indices, shp_3d)
            call c_f_pointer(mesh_5d_data%not_filler_ptr, &
                             this%not_filler, shp_3d)
            call c_f_pointer(mesh_5d_data%is_compute_ptr, &
                             this%is_compute, shp_3d)
            call c_f_pointer(mesh_5d_data%jacobian_buffer_ptr, &
                             this%jacobian_buffer, shp_3d)

            ! Swap the Fortran/C++ pointers of members related to the
            ! magnetic field
            call c_f_pointer(mesh_5d_data%absB_buffer_ptr, &
                             this%absB_buffer, shp_3d)
            call c_f_pointer(mesh_5d_data%normb_R_buffer_ptr, &
                             this%normb_R_buffer, shp_3d)
            call c_f_pointer(mesh_5d_data%normb_Z_buffer_ptr, &
                             this%normb_Z_buffer, shp_3d)
            call c_f_pointer(mesh_5d_data%curl_normb_y_ptr, &
                             this%curl_normb_y, shp_3d)
            call c_f_pointer(mesh_5d_data%dgyxdy_over_g_ptr, &
                             this%dgyxdy_over_g, shp_3d)
            call c_f_pointer(mesh_5d_data%dgyzdy_over_g_ptr, &
                             this%dgyzdy_over_g, shp_3d)
            call c_f_pointer(mesh_5d_data%dgyxdz_over_g_ptr, &
                             this%dgyxdz_over_g, shp_3d)
            call c_f_pointer(mesh_5d_data%dgyzdx_over_g_ptr, &
                             this%dgyzdx_over_g, shp_3d)
            call c_f_pointer(mesh_5d_data%inv_g_ptr, this%inv_g, shp_3d)
            call c_f_pointer(mesh_5d_data%dabsBdx_ptr, this%dabsBdx, shp_3d)
            call c_f_pointer(mesh_5d_data%dabsBdz_ptr, this%dabsBdz, shp_3d)
            call c_f_pointer(mesh_5d_data%dabsBdy_ptr, this%dabsBdy, shp_3d)

            ! Swap the Fortran/C++ pointers of members related to parallel
            ! connection
            call c_f_pointer(mesh_5d_data%fll_positive1_ptr, &
                             this%fll_positive1, shp_3d)
            call c_f_pointer(mesh_5d_data%fll_positive2_ptr, &
                             this%fll_positive2, shp_3d)
            call c_f_pointer(mesh_5d_data%fll_negative1_ptr, &
                             this%fll_negative1, shp_3d)
            call c_f_pointer(mesh_5d_data%fll_negative2_ptr, &
                             this%fll_negative2, shp_3d)
            call c_f_pointer(mesh_5d_data%not_in_target_ptr, &
                             this%not_in_target, shp_3d)

            ! Shift the lower bounds of members related to the mesh
            this%neighbor_indices(-size_neighbor:, -size_neighbor:, &
                                  1:, this%lb_phi:) => this%neighbor_indices
            this%buf_zone(1:, this%lb_phi:)         => this%buf_zone
            this%RZ_indices(1:, this%lb_phi:)       => this%RZ_indices
            this%not_filler(1:, this%lb_phi:)       => this%not_filler
            this%is_compute(1:, this%lb_phi:)       => this%is_compute
            this%jacobian_buffer(1:, this%lb_phi:)  => this%jacobian_buffer

            ! Shift the lower bounds of members related to the magnetic field
            this%absB_buffer(1:, this%lb_phi:)    => this%absB_buffer
            this%normb_R_buffer(1:, this%lb_phi:) => this%normb_R_buffer
            this%normb_Z_buffer(1:, this%lb_phi:) => this%normb_Z_buffer
            this%curl_normb_y(1:, this%lb_phi:)   => this%curl_normb_y
            this%dgyxdy_over_g(1:, this%lb_phi:)  => this%dgyxdy_over_g
            this%dgyzdy_over_g(1:, this%lb_phi:)  => this%dgyzdy_over_g
            this%dgyxdz_over_g(1:, this%lb_phi:)  => this%dgyxdz_over_g
            this%dgyxdz_over_g(1:, this%lb_phi:)  => this%dgyxdz_over_g
            this%inv_g(1:, this%lb_phi:)          => this%inv_g
            this%dabsBdx(1:, this%lb_phi:)        => this%dabsBdx
            this%dabsBdz(1:, this%lb_phi:)        => this%dabsBdz
            this%dabsBdy(1:, this%lb_phi:)        => this%dabsBdy

            ! Shift the lower bounds of members related to parallel connection
            this%fll_positive1(1:, this%lb_phi:) => this%fll_positive1
            this%fll_positive2(1:, this%lb_phi:) => this%fll_positive2
            this%fll_negative1(1:, this%lb_phi:) => this%fll_negative1
            this%fll_negative2(1:, this%lb_phi:) => this%fll_negative2
            this%not_in_target(1:, this%lb_phi:) => this%not_in_target
        end subroutine

    end subroutine

    module subroutine finalize_gpu(this)
        class(mesh_5d_t), intent(inout) :: this

        integer :: ierr
        ! C++ error status

        ierr = cbind_mesh_5d_finalize(this%mesh_5d_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
