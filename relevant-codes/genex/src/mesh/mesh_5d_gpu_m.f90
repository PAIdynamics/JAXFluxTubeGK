module mesh_5d_gpu_m
    !! Module containing additional resources related to GPU and Fortran/C++
    !! interoperability for the mesh_5d_t type
    use, intrinsic :: iso_c_binding, only: C_PTR, C_INT32_T, C_CHAR
    use genex_fortran_env_m, only: CP

    implicit none
    private

    public :: cbind_mesh_5d_initialize
    public :: cbind_mesh_5d_finalize

    type, public, bind(C) :: mesh_5d_data_t
        !! Fortran/C++ interoperable structure of the class members
        !! relevant to mesh_5d_t

        integer(C_INT32_T) :: size_RZ
        !! Maximum number of points in RZ direction including ghost
        !! and filler points
        integer(C_INT32_T) :: size_phi
        !! Number of points in phi direction
        integer(C_INT32_T) :: size_vp
        !! Number of points in vp direction
        integer(C_INT32_T) :: size_mu
        !! Number of points in mu direction
        integer(C_INT32_T) :: size_sp
        !! Number of points in sp direction
        integer(C_INT32_T) :: order_RZ_stencil
        !! Number of neighboring points in each direction (left, right, bottom,
        !! top) or the order of RZ stencil
        integer(C_INT32_T) :: lb_phi
        !! Lower bound on the index of phi planes
        integer(C_INT32_T) :: ub_phi
        !! Upper bound on the index of phi planes
        real(kind=CP) :: delta_RZ
        !! Grid spacing of the RZ grid (same for every plane)
        real(kind=CP) :: delta_phi
        !! Grid spacing of the toroidal angle grid
        real(kind=CP) :: delta_vp
        !! Grid spacing of the parallel velocity grid
        real(kind=CP) :: delta_sqrt_mu
        !! Grid spacing of the magnetic moment grid at the first
        !! point of the grid. If the mu grid is quadratic, this spacing
        !! will be the same for all points.
        type(C_PTR) :: neighbors_ptr
        !! C pointer to the indices of the neighbor points
        type(C_PTR) :: buf_zone_ptr
        !! C pointer to an array specifying which buffer zone a given point is
        !! in, identified by the buffer zone type enumerator
        type(C_PTR) :: RZ_indices_ptr
        !! C pointer to the mesh grid indices in the RZ dimension
        type(C_PTR) :: not_filler_ptr
        !! C pointer to a mask array indicating whether a buffer point is a
        !! filler point, for meshes which have fewer points than the size of the
        !! array. The value is 0 for filler points, otherwise the value is 1.
        type(C_PTR) :: is_compute_ptr
        !! C pointer to the array specifying whether a given point is in the
        !! computation domain
        type(C_PTR) :: vp_grid_ptr
        !! C pointer to parallel velocity grid
        type(C_PTR) :: mu_grid_ptr
        !! C pointer to mu grid
        type(C_PTR) :: sqrt_mu_grid_ptr
        !! C pointer to sqrt(mu) grid
        type(C_PTR) :: vp_weights_ptr
        !! C pointer to the integration weights of the vp grid
        type(C_PTR) :: mu_weights_ptr
        !! C pointer to the integration weights of the mu grid
        type(C_PTR) :: jacobian_buffer_ptr
        !! C pointer to the jacobian of the coordinate transform from
        !! lab coordinates into the coordinate system of the mesh

        ! Pointers related to magnetic field (magfield)

        type(C_PTR) :: absB_buffer_ptr
        !! C pointer to B field on the 3D mesh
        type(C_PTR) :: normb_R_buffer_ptr
        !! C pointer to R component of the normalized magnetic field
        type(C_PTR) :: normb_Z_buffer_ptr
        !! C pointer to Z component of the normalized magnetic field
        type(C_PTR) :: curl_normb_y_ptr
        !! C pointer to y component of the curl of the normalized magnetic
        !! field
        type(C_PTR) :: dgyxdy_over_g_ptr
        !! C pointer to the derivative of b_R along the magnetic field on
        !! the 3D mesh
        type(C_PTR) :: dgyzdy_over_g_ptr
        !! C pointer to the derivative of b_Z along the magnetic field on
        !! the 3D mesh
        type(C_PTR) :: dgyxdz_over_g_ptr
        !! C pointer the derivative of b_R in Z direction on the 3D mesh
        type(C_PTR) :: dgyzdx_over_g_ptr
        !! C pointer to the derivative of b_Z in R direction on the 3D mesh
        type(C_PTR) :: inv_g_ptr
        !! C pointer to the inverse of sqrt(abs(det(g))) where g denotes
        !! the metric tensor of the x, y, z coordinate system
        type(C_PTR) :: dabsBdx_ptr
        !! C pointer to the derivative of abs B in R direction on the 3D mesh
        type(C_PTR) :: dabsBdz_ptr
        !! C pointer to the derivative of abs B in Z direction on the 3D mesh
        type(C_PTR) :: dabsBdy_ptr
        !! C pointer to the derivative of abs B in y direction on the 3D mesh

        ! Pointers related to parallel connection (parcon)

        type(C_PTR) :: map_positive1_data_ptr
        !! C pointer to the map matrix structures for the poloidal planes k + 1
        type(C_PTR) :: map_negative1_data_ptr
        !! C pointer to the map matrix structures for the poloidal planes k - 1
        type(C_PTR) :: map_positive2_data_ptr
        !! C pointer to the map matrix structures for the poloidal planes k + 2
        type(C_PTR) :: map_negative2_data_ptr
        !! C pointer to the map matrix structures for the poloidal planes k - 2
        type(C_PTR) :: fll_positive1_ptr
        !! C pointer to field line lengths from each point to the k + 1 plane
        type(C_PTR) :: fll_positive2_ptr
        !! C pointer to field line lengths from each point to the k + 2 plane
        type(C_PTR) :: fll_negative1_ptr
        !! C pointer to field line lengths from each point to the k - 1 plane
        type(C_PTR) :: fll_negative2_ptr
        !! C pointer to field line lengths from each point to the k - 2 plane
        type(C_PTR) :: not_in_target_ptr
        !! C pointer to the array specifying whether a given point is not in
        !! the target
    end type

    interface
        integer(kind=C_INT32_T) function cbind_mesh_5d_initialize( &
            mesh_data, grid_type_phi, grid_type_vp, grid_type_mu, &
            quad_type_phi, quad_type_vp, quad_type_mu, mesh_cxx_pptr) &
            bind(C, name="cbind_mesh_5d_initialize")
            !! Fortran/C++ interoperable routine for the initialization of
            !! the mesh_5d_t C++ class
            import :: C_INT32_T, C_PTR, C_CHAR, mesh_5d_data_t
            type(mesh_5d_data_t), intent(inout) :: mesh_data
            !! Object of Fortran/C++ interoperable
            !! structure based on mesh_5d_data_t
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: grid_type_phi
            !! C characters specifying the type of phi grid
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: grid_type_vp
            !! C characters specifying the type of vp grid
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: grid_type_mu
            !! C characters specifying the type of mu grid
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: quad_type_phi
            !! C characters specifying the type of phi quad
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: quad_type_vp
            !! C characters specifying the type of vp quad
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: quad_type_mu
            !! C characters specifying the type of mu quad
            type(C_PTR), intent(inout) :: mesh_cxx_pptr
            !! C pointer to the mesh_5d_t C++ class instance pointer
        end function

        integer(kind=C_INT32_T) function cbind_mesh_5d_finalize( &
            mesh_cxx_pptr) bind(C, name="cbind_mesh_5d_finalize")
            !! Fortran/C++ interoperable routine for the finalization of
            !! the mesh_5d_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: mesh_cxx_pptr
            !! C pointer to the mesh_5d_t C++ class instance pointer
        end function
    end interface

end module
