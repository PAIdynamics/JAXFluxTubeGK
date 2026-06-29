module helpers_mesh_5d_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of mesh_5d_t
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_PTR, C_CHAR
    use mesh_5d_gpu_m, only: mesh_5d_data_t

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_mesh_copy_scalar( &
            mesh_src_cxx_ptr, mesh_data_tgt, grid_type_phi, grid_type_vp, &
            grid_type_mu, quad_type_phi, quad_type_vp, quad_type_mu) &
            bind(C, name="cbind_mesh_copy_scalar")
            !! Interoperable routine to copy the scalar members from the source
            !! mesh to target mesh struct
            import :: C_INT32_T, C_PTR, C_CHAR, mesh_5d_data_t
            type(C_PTR), intent(in)             :: mesh_src_cxx_ptr
            type(mesh_5d_data_t), intent(inout) :: mesh_data_tgt
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: grid_type_phi
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: grid_type_vp
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: grid_type_mu
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: quad_type_phi
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: quad_type_vp
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: quad_type_mu
        end function

        integer(kind=C_INT32_T) function cbind_mesh_copy_array( &
            mesh_src_cxx_ptr, mesh_data_tgt) &
            bind(C, name="cbind_mesh_copy_array")
            !! Interoperable routine to copy the array members from the source
            !! mesh to target mesh struct
            import :: C_INT32_T, C_PTR, mesh_5d_data_t
            type(C_PTR), intent(in)             :: mesh_src_cxx_ptr
            type(mesh_5d_data_t), intent(inout) :: mesh_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_mesh_copy_magfield( &
            mesh_src_cxx_ptr, mesh_data_tgt) &
            bind(C, name="cbind_mesh_copy_magfield")
            !! Interoperable routine to copy the class members from the source
            !! mesh to target mesh struct that are related to the magnetic field
            !! (magfield)
            import :: C_INT32_T, C_PTR, mesh_5d_data_t
            type(C_PTR), intent(in)             :: mesh_src_cxx_ptr
            type(mesh_5d_data_t), intent(inout) :: mesh_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_mesh_copy_parcon( &
            mesh_src_cxx_ptr, mesh_data_tgt) &
            bind(C, name="cbind_mesh_copy_parcon")
            !! Interoperable routine to copy the class members related to
            !! the parallel connection (parcon) from the source mesh to
            !! target mesh struct
            import :: C_INT32_T, C_PTR, mesh_5d_data_t
            type(C_PTR), intent(in)             :: mesh_src_cxx_ptr
            type(mesh_5d_data_t), intent(inout) :: mesh_data_tgt
        end function

    end interface

end module
