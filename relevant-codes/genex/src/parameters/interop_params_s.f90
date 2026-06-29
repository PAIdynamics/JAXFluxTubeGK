submodule(interop_params_m) interop_params_s
    !! Submodule containing routines and other resources to interoprate
    !! the parameters between Fortran and C++
    use, intrinsic :: iso_c_binding, only: C_CHAR
    use genex_fortran_env_m, only: GP
    use type_converters_m, only: string_f2c
    use params_gpu_offload_m
    use params_collisions_m
    use params_gyrokinetic_system_m
    use params_normalization_m
    use params_numerical_scheme_m
    use params_devtools_m
    use params_species_m

    implicit none

#ifdef ENABLE_GPU
    interface
        subroutine cbind_set_params_gpu_offload(params_data) &
            bind(C, name="cbind_set_params_gpu_offload")
            import :: params_gpu_offload_data_t
            !! Fortran/C++ interoperable routine for initialization of
            !! params_gpu_offload on C++ layer
            type(params_gpu_offload_data_t) :: params_data
        end subroutine

        subroutine cbind_set_params_collisions(coll_type, relx_type, &
            dens_floor, temp_floor) bind(C, name="cbind_set_params_collisions")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_collisions on C++ layer
            import :: C_CHAR, CP
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: coll_type
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: relx_type
            real(kind=CP), value :: dens_floor
            real(kind=CP), value :: temp_floor
        end subroutine

        subroutine cbind_set_params_devtools(isolate_tsync, trace_dmem) &
            bind(C, name="cbind_set_params_devtools")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_devtools on C++ layer
            import :: C_INT32_T
            integer(kind=C_INT32_T), value :: isolate_tsync
            integer(kind=C_INT32_T), value :: trace_dmem
        end subroutine

        subroutine cbind_set_params_gyrokinetic_system(with_rhs, &
            with_em_flutter, with_nlin_polarization, with_bpar) &
            bind(C, name="cbind_set_params_gyrokinetic_system")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_gyrokinetic_system on C++ layer
            import :: C_INT32_T
            integer(kind=C_INT32_T), value :: with_rhs
            integer(kind=C_INT32_T), value :: with_em_flutter
            integer(kind=C_INT32_T), value :: with_nlin_polarization
            integer(kind=C_INT32_T), value :: with_bpar
        end subroutine

        subroutine cbind_set_params_species(masses_ptr, charges_ptr, &
                                            temp_scalings_ptr) &
            bind(C, name="cbind_set_params_species")
            !! Fortran/C++ interoperable routine for initialization of
            !! params_species on C++ layer
            import :: CP
            real(kind=CP), dimension(*), intent(in) :: masses_ptr
            real(kind=CP), dimension(*), intent(in) :: charges_ptr
            real(kind=CP), dimension(*), intent(in) :: temp_scalings_ptr
        end subroutine

        subroutine cbind_set_params_normalization(params_data) &
            bind(C, name="cbind_set_params_normalization")
            import :: params_normalization_data_t
            !! Fortran/C++ interoperable routine for initialization of
            !! params_normalization on C++ layer
            type(params_normalization_data_t) :: params_data
        end subroutine

        subroutine cbind_set_params_numerical_scheme(params_data) &
            bind(C, name="cbind_set_params_numerical_scheme")
            import :: params_numerical_scheme_data_t
            !! Fortran/C++ interoperable routine for initialization of
            !! params_normalization on C++ layer
            type(params_numerical_scheme_data_t) :: params_data
        end subroutine
    end interface
#endif

contains

    module subroutine set_cxx_params_gpu_offload()

        type(params_gpu_offload_data_t) :: params_data

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            params_data%use_gpu_offload = 1
            params_data%gpu_offload_backend = get_gpu_offload_backend()
            params_data%use_parallax_gpu_offload = &
                merge(1, 0, get_use_parallax_gpu_offload())
            params_data%parallax_gpu_offload_backend = &
                get_parallax_gpu_offload_backend()
            params_data%use_parallax_gpu_data_explicit = &
                merge(1, 0, get_use_parallax_gpu_data_explicit())
            params_data%swap_mesh_members = merge(1, 0, get_swap_mesh_members())
            params_data%large_array_alignment = get_array_alignment()

            call cbind_set_params_gpu_offload(params_data)
        endif
#endif
    end subroutine

    module subroutine set_cxx_params_collisions()

        character(len=1, kind=C_CHAR), dimension(16) :: coll_type_c
        ! Collision type in C character array format
        character(len=1, kind=C_CHAR), dimension(16) :: relx_type_c
        ! Relaxation type in C character array format

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Convert Fortran-formatted strings to C character array format
            call string_f2c(get_coll_type(), coll_type_c)
            call string_f2c(get_relx_type(), relx_type_c)

            call cbind_set_params_collisions(coll_type_c, relx_type_c, &
                                             get_dens_floor(), get_temp_floor())
        endif
#endif
    end subroutine

    module subroutine set_cxx_params_devtools()
#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            call cbind_set_params_devtools( &
                merge(1, 0, get_isolate_tsync()), &
                merge(1, 0, get_trace_device_memory()))
        endif
#endif
    end subroutine

    module subroutine set_cxx_params_gyrokinetic_system()
#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            call cbind_set_params_gyrokinetic_system( &
                merge(1, 0, get_with_rhs()), &
                merge(1, 0, get_with_em_flutter()), &
                merge(1, 0, get_with_nlin_polarization()), &
                merge(1, 0, get_with_bpar()))

        endif
#endif
    end subroutine

    module subroutine set_cxx_params_normalization()

        type(params_normalization_data_t) :: params_data

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            params_data%m_ref    = get_m_ref()
            params_data%T_ref    = get_T_ref()
            params_data%n_ref    = get_n_ref()
            params_data%L_ref    = get_L_ref()
            params_data%B_ref    = get_B_ref()
            params_data%c_ref    = get_c_ref()
            params_data%rho_ref  = get_rho_ref()
            params_data%coll_ref = get_coll_ref()
            params_data%beta_ref = get_beta_ref()

            call cbind_set_params_normalization(params_data)
        endif
#endif
    end subroutine

    module subroutine set_cxx_params_numerical_scheme()

        type(params_numerical_scheme_data_t) :: params_data

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            params_data%int_order = get_int_order()
            params_data%hyp_y     = get_hyp_y()
            params_data%hyp_xz    = get_hyp_xz()
            params_data%hyp_vp    = get_hyp_vp()
            params_data%buf_zone_size          = get_buf_zone_size()
            params_data%buf_zone_strength      = get_buf_zone_strength()
            params_data%buf_zone_size_axis     = get_buf_zone_size_axis()
            params_data%buf_zone_strength_axis = get_buf_zone_strength_axis()

            call cbind_set_params_numerical_scheme(params_data)
        endif
#endif
    end subroutine

    module subroutine set_cxx_params_species()

        real(kind=GP), dimension(n_spec_supported) :: masses, &
                                                      charges, &
                                                      temp_scalings
        integer :: n

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            do n = 1, n_spec_supported
                masses(n)  = get_mass(n)
                charges(n) = get_charge(n)
                temp_scalings(n) = get_temp_scaling(n)
            enddo
            call cbind_set_params_species(masses, charges, temp_scalings)

        endif
#endif
    end subroutine

end submodule
