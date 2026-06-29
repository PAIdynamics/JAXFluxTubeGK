module interop_params_m
    !! Module containing routines and other resources to interoprate
    !! the parameters between Fortran and C++
    use, intrinsic :: iso_c_binding, only: C_INT32_T
    use genex_fortran_env_m, only: CP

    implicit none

    interface
        module subroutine set_cxx_params_gpu_offload()
            !! Sets params_gpu_offload to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_collisions()
            !! Sets params_collisions to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_gyrokinetic_system()
            !! Sets params_gyrokinetic_system to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_normalization()
            !! Sets params_normalization to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_numerical_scheme()
            !! Sets params_numerical_scheme to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_devtools()
            !! Sets params_devtools to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine

        module subroutine set_cxx_params_species()
            !! Sets params_species to C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine
    end interface

    type, public, bind(C) :: params_gpu_offload_data_t
        !! Fortran/C++ interoperable structure/container of parameter objects
        !! relevant to the GPU offloading
        integer(kind=C_INT32_T) :: use_gpu_offload
        integer(kind=C_INT32_T) :: gpu_offload_backend
        integer(kind=C_INT32_T) :: use_parallax_gpu_offload
        integer(kind=C_INT32_T) :: parallax_gpu_offload_backend
        integer(kind=C_INT32_T) :: use_parallax_gpu_data_explicit
        integer(kind=C_INT32_T) :: swap_mesh_members
        integer(kind=C_INT32_T) :: large_array_alignment
    end type

    type, public, bind(C) :: params_normalization_data_t
        !! Fortran/C++ interoperable structure/container of parameter objects
        !! relevant to the normalization
        real(kind=CP) :: m_ref
        real(kind=CP) :: T_ref
        real(kind=CP) :: n_ref
        real(kind=CP) :: L_ref
        real(kind=CP) :: B_ref
        real(kind=CP) :: c_ref
        real(kind=CP) :: rho_ref
        real(kind=CP) :: coll_ref
        real(kind=CP) :: beta_ref
    end type

    type, public, bind(C) :: params_numerical_scheme_data_t
        !! Fortran/C++ interoperable structure/container of parameter objects
        !! relevant to the numerical scheme
        integer(kind=C_INT32_T) :: int_order
        real(kind=CP)           :: hyp_y
        real(kind=CP)           :: hyp_xz
        real(kind=CP)           :: hyp_vp
        integer(kind=C_INT32_T) :: buf_zone_size
        real(kind=CP)           :: buf_zone_strength
        integer(kind=C_INT32_T) :: buf_zone_size_axis
        real(kind=CP)           :: buf_zone_strength_axis
    end type

end module
