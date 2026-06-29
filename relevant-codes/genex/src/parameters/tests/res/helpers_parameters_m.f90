module helpers_parameters_m
    !! Contains Fortran/C++ interface for testing the C++ features
    !! of the parameters

    use, intrinsic :: iso_c_binding, only: C_CHAR, C_INT32_T
    use genex_fortran_env_m, only: CP
    use interop_params_m, only: params_gpu_offload_data_t, &
                                params_normalization_data_t, &
                                params_numerical_scheme_data_t

    implicit none

    interface
        module subroutine cbind_get_params_collisions(scalar_params, &
            coll_type, relx_type) bind(C, name="cbind_get_params_collisions")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_collisions on C++ layer
            real(kind=CP), dimension(*), intent(inout) :: scalar_params
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: coll_type
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(inout) :: relx_type
        end subroutine

        module subroutine cbind_get_params_devtools(scalar_params) &
            bind(C, name="cbind_get_params_devtools")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_devtools on C++ layer
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: &
                scalar_params
        end subroutine

        module subroutine cbind_get_params_gyrokinetic_system(scalar_params) &
            bind(C, name="cbind_get_params_gyrokinetic_system")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_gyrokinetic_system on C++ layer
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: &
                scalar_params
        end subroutine

        module subroutine cbind_get_params_gpu_offload(params_data) &
            bind(C, name="cbind_get_params_gpu_offload")
            !! Interoperable routine to get the GPU offload related parameters
            !! from C++ layer
            type(params_gpu_offload_data_t), intent(inout) :: params_data
        end subroutine

        module subroutine cbind_get_params_normalization(params_data) &
            bind(C, name="cbind_get_params_normalization")
            !! Interoperable routine to get the normalization-related parameters
            !! from C++ layer
            type(params_normalization_data_t), intent(inout) :: params_data
        end subroutine

        module subroutine cbind_get_params_numerical_scheme(params_data) &
            bind(C, name="cbind_get_params_numerical_scheme")
            !! Interoperable routine to get the numerical_scheme-related
            !! parameters from C++ layer
            type(params_numerical_scheme_data_t), intent(inout) :: params_data
        end subroutine

        module subroutine cbind_get_params_species(n_spec, electron_id, &
                                                   masses_ptr, charges_ptr, &
                                                   temp_scalings_ptr) &
            bind(C, name="cbind_get_params_species")
            !! Interoperable routine to get the species-related parameters
            !! from C++ layer
            integer(kind=C_INT32_T), value :: n_spec
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: &
                electron_id
            real(kind=CP), dimension(*), intent(inout) :: masses_ptr
            real(kind=CP), dimension(*), intent(inout) :: charges_ptr
            real(kind=CP), dimension(*), intent(inout) :: temp_scalings_ptr
        end subroutine
    end interface

end module
