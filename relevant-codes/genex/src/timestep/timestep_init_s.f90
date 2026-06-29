submodule (timestep_m) timestep_init_s
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS, &
                                    GENEX_ERR_TIMESTEP, &
                                    GENEX_WRN_FILE_NOT_EXIST, &
                                    GENEX_WRN_PARAMETERS, GENEX_ERR_BSG
    use logger_m, only: logger_get_info_channel
    use type_converters_m, only: string
    use params_mesh_m, only: get_n_points_sp, get_use_vspectral, get_use_bsg, &
                             get_use_vspectral
    use params_bsg_m, only: get_num_bsg_blocks, get_bsg_interp_order
    use params_species_m, only: get_charge, get_name, get_is_electrons
    use params_species_config_m, only: get_params_species_config, &
                                       params_species_config_t
    use params_neutrals_m, only: get_n_points_neut, get_neut_mapped_ion_name
    use params_neutrals_config_m, only: get_neut_evolve_type, &
                                        get_neut_coupling_type, &
                                        get_use_neutrals
    use params_normalization_m, only: get_n_ref, get_T_ref, get_rate_ref
    use params_profile_cbc_m, only:  get_params_profile_cbc_dens, &
                                     get_params_profile_cbc_temp, &
                                     get_params_profile_cbc_temp_par, &
                                     get_params_profile_cbc_temp_perp
    use params_profile_sine_m, only: get_params_profile_sine_dens, &
                                     get_params_profile_sine_temp, &
                                     get_params_profile_sine_temp_par, &
                                     get_params_profile_sine_temp_perp
    use params_profile_lapd_m, only: get_params_profile_lapd_dens, &
                                     get_params_profile_lapd_temp
    use params_gyrokinetic_system_m, only: get_with_rhs, get_with_bpar
    use params_numerical_scheme_m, only: get_buf_zone_size
    use params_profile_cbc_m, only: params_profile_cbc_t
    use params_profile_sine_m, only: params_profile_sine_t
    use params_profile_lapd_m, only: params_profile_lapd_t
    use params_gpu_offload_m, only: get_use_gpu_offload
    use params_diagnostics_m, only: get_diagnose_tpc
    use profiles_m, only: profile_t, profile_cbc_cpu_t, profile_lapd_cpu_t, &
                          profile_sine_cpu_t, profile_uniform_cpu_t
    use dist_initial_m, only: dist_initial_t, dist_initial_maxw_cpu_t, &
                              dist_initial_double_maxw_cpu_t, &
                              dist_initial_ring_cpu_t, &
                              dist_initial_slowing_down_cpu_t, &
                              dist_initial_bi_maxw_cpu_t, &
                              dist_initial_maxw_vspec_cpu_t, &
                              dist_initial_maxw_loc_vspec_cpu_t
    use op_set_initial_condition_m, only: op_set_initial_condition_cpu_t, &
                                          op_set_initial_condition_mms_cpu_t, &
                                          op_set_initial_condition_vspec_cpu_t
    use op_neut_set_initial_m, only: op_neut_set_initial_cpu_t
    use op_solve_qn_eq_m, only: op_solve_qn_eq_t
    use op_solve_amps_law_m, only: op_solve_amps_law_t
    use op_solve_bpar_eq_m, only: op_solve_bpar_eq_t
    use op_solve_ohms_law_m, only: op_solve_ohms_law_t
    use op_mms_source_m, only: op_mms_source_vlasov_eq_cpu_t, &
                               op_mms_source_maxwells_eq_cpu_t, &
                               op_mms_source_ohms_law_cpu_t
    use op_mms_error_m, only: op_mms_error_cpu_t
    use op_mms_solution_m, only: op_mms_solution_vlasov_eq_cpu_t, &
                                 op_mms_solution_maxwells_eq_cpu_t, &
                                 op_mms_solution_ohms_law_cpu_t
    use op_neut_mom_m, only: op_neut_mom_cpu_t
    use op_neut_evolve_m, only: op_neut_evolve_dummy_t, &
                                op_neut_evolve_1mom_t, &
                                op_neut_evolve_1mom_2cfd_t, &
                                op_neut_evolve_1mom_diff_t, &
                                op_neut_evolve_1mom_4cfd_t
    use op_neut_coupling_np_m, only: op_neut_coupling_np_none_t, &
                                     op_neut_coupling_np_krook_t
    use op_neut_coupling_pn_m, only: op_neut_coupling_pn_none_t, &
                                     op_neut_coupling_pn_temp_t, &
                                     op_neut_coupling_pn_krook_t
    use part_handler_m, only: inquire_part, create_part
    use bsg_operators_m, only: bsg_operators_t

    ! From REAX
    use rate_coefficients_m, only: &
                               reax_initialize_rate_norm => initialize_rate_norm

    implicit none

contains

    module subroutine initialize_parent(this, dcomm_handler, mesh)
        class(timestep_t), target, intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        type(mesh_5d_t), pointer, intent(in) :: mesh

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        ! TODO_BSG: Expand this implementation to GPU in the future
        allocate(bsg_operators_t :: this%bsg_op)
        if(get_use_bsg()) then
            call this%bsg_op%initialize(this%mesh, get_num_bsg_blocks(), &
                                        get_bsg_interp_order())
        else
            call this%bsg_op%initialize(this%mesh, 1)
        end if
        if(.not. this%bsg_op%is_initialized()) then
            call handle_error("BSG operator object not initialized", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        this%apply_plasma = get_with_rhs()

        ! NOTE: The order of the following initializations is important. The
        !       state vector determines the number of ghost cells from the
        !       allocated operators.
        call this%check_species_config()
        call this%initialize_profiles()
        ! NOTE: Init of dist_initial needs profiles initialized before
        call this%initialize_dist_initial()
        if(get_use_neutrals()) call this%check_neutrals_config()
        if(.not. this%run_test) call this%initialize_operators()
        call this%initialize_state_vector()
        call this%initialize_diagnostics()

        ! Make a check if buffer zone was accidentally set to zero
        if(get_buf_zone_size() == 0 .and. .not. this%run_mms) then
            call handle_error("Buffer zone size was set to zero. If this &
                              &was unintentional, check if the parameter &
                              &is indeed an integer and not a real!", &
                              GENEX_WRN_PARAMETERS, __LINE__, __FILE__)
        end if

    end subroutine

    module subroutine initialize_profiles(this)
        class(timestep_t), intent(inout) :: this

        integer :: n, n_spec
        logical :: is_isotropic

        class(profile_t), allocatable :: profile_dens, profile_temp, &
                                         profile_temp_par, profile_temp_perp
        type(params_species_config_t) :: psc

        ! NOTE: For the MMS, the profiles are not used, as
        !       the initial condition is set using the solution file.
        if(this%run_mms) return

        if(this%dcomm_handler%is_master() .and. .not. this%run_mms) then
            write(logger_get_info_channel(), *) "   Initialize profiles"
        end if

        ! Initialize profile container
        n_spec = get_n_points_sp()
        allocate(this%profile_container(n_spec))
        do n = 1, n_spec

            psc = get_params_species_config(n)
            is_isotropic = psc%is_isotropic

            call alloc_profile(psc%profile_type_dens, "dens", &
                               n, profile_dens)
            call alloc_profile(psc%profile_type_temp, "temp", &
                               n, profile_temp)
            call alloc_profile(psc%profile_type_temp_par, "temp_par", &
                               n, profile_temp_par)
            call alloc_profile(psc%profile_type_temp_perp, "temp_perp",&
                               n, profile_temp_perp)

            call this%profile_container(n)%initialize(is_isotropic, &
                        profile_dens, profile_temp, profile_temp_par, &
                        profile_temp_perp)
        end do

    contains
        subroutine alloc_profile(profname, typename, n, profile)
            !! This subroutine is a helper to initialize a single profile
            !! given by the name (from parameter), the typename (from code),
            !! the species index and a polymorphic type that should be allocated
            character(len=*), intent(in) :: profname
            character(len=*), intent(in) :: typename
            integer, intent(in) :: n
            class(profile_t), allocatable, intent(inout) :: profile

            logical :: is_density, is_electrons

            type(profile_uniform_cpu_t), allocatable :: profile_uniform
            type(profile_cbc_cpu_t),     allocatable :: profile_cbc
            type(profile_sine_cpu_t),    allocatable :: profile_sine
            type(profile_lapd_cpu_t),    allocatable :: profile_lapd

            type(params_profile_cbc_t)  :: params_cbc
            type(params_profile_sine_t) :: params_sine
            type(params_profile_lapd_t) :: params_lapd

            is_electrons = get_is_electrons(n)

            ! Initialize correct profile
            select case(profname)
                case("uniform")
                    allocate(profile_uniform)
                    call move_alloc(profile_uniform, profile)

                case("cbc")
                    allocate(profile_cbc)

                    select case(typename)
                        case("dens")
                            params_cbc = get_params_profile_cbc_dens(n)
                        case("temp")
                            params_cbc = get_params_profile_cbc_temp(n)
                        case("temp_par")
                            params_cbc = get_params_profile_cbc_temp_par(n)
                        case("temp_perp")
                            params_cbc = get_params_profile_cbc_temp_perp(n)
                    end select

                    call profile_cbc%initialize(params_cbc)
                    call move_alloc(profile_cbc, profile)

                case("sine")
                    allocate(profile_sine)

                    select case(typename)
                        case("dens")
                            params_sine=get_params_profile_sine_dens(n)
                        case("temp")
                            params_sine=get_params_profile_sine_temp(n)
                        case("temp_par")
                            params_sine=get_params_profile_sine_temp_par(n)
                        case("temp_perp")
                            params_sine=get_params_profile_sine_temp_perp(n)
                    end select

                    call profile_sine%initialize(params_sine)
                    call move_alloc(profile_sine, profile)

                case("lapd")
                    allocate(profile_lapd)

                    ! We need information if the profile is a density or not
                    ! for the lapd init
                    select case(typename)
                        case("dens")
                            params_lapd = get_params_profile_lapd_dens(n)
                            is_density = .true.
                        case("temp")
                            params_lapd = get_params_profile_lapd_temp(n)
                            is_density = .false.
                        ! For par and perp temp we also use the normal
                        ! temp since lapd only has isotropic temp
                        case("temp_par")
                            params_lapd = get_params_profile_lapd_temp(n)
                            is_density = .false.
                        case("temp_perp")
                            params_lapd = get_params_profile_lapd_temp(n)
                            is_density = .false.
                    end select

                    call profile_lapd%initialize(this%mesh, params_lapd, &
                                                 is_density, is_electrons)
                    call move_alloc(profile_lapd, profile)

                case default
                    call handle_error(&
                            "Selected initial profile is not supported!", &
                            GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                            additional_info=&
                                error_info_t("Profile was: "//profname))
            end select
        end subroutine
    end subroutine

    module subroutine initialize_dist_initial(this)
        class(timestep_t), intent(inout) :: this
        class(dist_initial_t), allocatable :: dist_initial
        integer :: n, n_spec
        type(params_species_config_t) :: params

        ! For MMS the dist_initial operator is not used, since the initial
        ! condition is set using the solution file
        if(this%run_mms) return

        ! Initialize profile container
        n_spec = get_n_points_sp()
        allocate(this%dist_initial_container(n_spec))
        do n = 1, n_spec

            ! Get params species config to fetch dist initial type char
            params = get_params_species_config(n)

            ! Initialize correct dist initial type
            select case(params%dist_initial_type)
                case("maxw")
                    allocate(dist_initial_maxw_cpu_t :: dist_initial)
                case("double_maxw")
                    allocate(dist_initial_double_maxw_cpu_t :: dist_initial)
                case("ring")
                    allocate(dist_initial_ring_cpu_t :: dist_initial)
                case("slowing_down")
                    allocate(dist_initial_slowing_down_cpu_t :: &
                                                              dist_initial)
                case("bi_maxw")
                    allocate(dist_initial_bi_maxw_cpu_t :: dist_initial)
                case('maxw_vspec')
                    allocate(dist_initial_maxw_vspec_cpu_t :: dist_initial)
                case('maxw_loc_vspec')
                    allocate(dist_initial_maxw_loc_vspec_cpu_t :: &
                                                                dist_initial)
                case default
                    call handle_error(&
                            "Selected initial distribution is not supported!", &
                            GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                            additional_info=&
                                error_info_t("Distribution was: "&
                                             &//params%dist_initial_type))
            end select

            call dist_initial%initialize(this%profile_container, n)

            ! Add to dist_initial container
            call this%dist_initial_container(n)%initialize(dist_initial)

            deallocate(dist_initial)
        end do

    end subroutine

    module subroutine check_species_config(this)
        class(timestep_t), intent(inout) :: this

        integer :: n, n_spec, n_electrons, n_ions_neg, n_ions_pos
        real(kind=GP) :: charge

        integer :: ierr
        character(len=:), allocatable :: err_message
        type(error_info_t) :: err_info

        if(this%dcomm_handler%is_master()) then
           write(logger_get_info_channel(), *) &
               "   Check plasma species configuration"
        end if

        ierr        = GENEX_SUCCESS
        n_spec      = get_n_points_sp()
        n_electrons = 0
        n_ions_neg  = 0
        n_ions_pos  = 0

        ! General Checks
        ! Require minimum 2 species
        if(n_spec < 2) then
            err_message = "Less than two species found!"
            err_info = error_info_t("Number of species was:", [n_spec])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        end if

        ! Count electrons and ions (pos + neg)
        do n = 1, n_spec
            charge = get_charge(n)
            if(charge == -1) n_electrons = n_electrons + 1
            if(charge /= -1 .and. charge < 0) n_ions_neg = n_ions_neg + 1
            if(charge > 0) n_ions_pos = n_ions_pos + 1
        end do

        ! Require exactly 1 electron species
        if(n_electrons == 0) then
            err_message = "No electron species found!"
            err_info = error_info_t("Number of electrons was:", [n_electrons])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        else if(n_electrons > 1) then
            err_message = "More than one electron species found!"
            err_info = error_info_t("Number of electrons was:", [n_electrons])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        end if

        ! Require no negative ion species
        if(n_ions_neg > 0) then
            err_message = "Negative ion species found!"
            err_info = error_info_t("Number of negative ions was:", &
                                    [n_ions_neg])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        end if

        ! Require minimum 1 positive ion species
        if(n_ions_pos < 1) then
            err_message = "No ion species found!"
            err_info = error_info_t("Number of ions was:", [n_ions_pos])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        end if

        ! Additional checks for MMS runs
        if(.not. this%run_mms) return

        ! Require exactly 2 species
        if(n_spec /= 2) then
            err_message = "Number of species for MMS not two!"
            err_info = error_info_t("Number of species was:", [n_spec])
            ierr = GENEX_ERR_PARAMETERS
            goto 10
        end if

        ! Require names are ions and electrons
        do n = 1, n_spec
            if(get_name(n) /= "ions" .and. get_name(n) /= "electrons") then
                err_message = "Name of species for MMS not ions or electrons!"
                err_info = error_info_t("Name of species was: "//get_name(n))
                ierr = GENEX_ERR_PARAMETERS
                goto 10
            end if
        end do

10      if(ierr /= GENEX_SUCCESS) then
            call handle_error(err_message, ierr, __LINE__, __FILE__, &
                              additional_info=err_info)
        end if

    end subroutine

    module subroutine check_neutrals_config(this)
        class(timestep_t), intent(inout) :: this

        integer :: n, o, n_neut, n_spec, n_ion_per_neut
        character(len=24), dimension(:), allocatable :: plasma_species_names
        character(len=24), dimension(:), allocatable :: mapped_ions_names

        integer :: ierr
        character(len=:), allocatable :: err_message
        type(error_info_t) :: err_info

        if(this%dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) &
                "   Check neutrals species configuration"
        end if

        ierr   = GENEX_SUCCESS
        n_spec = get_n_points_sp()
        n_neut = get_n_points_neut()
        allocate(plasma_species_names(n_spec))
        allocate(mapped_ions_names(n_neut))

        ! Require that the number of neutrals species
        ! is lower or equal of the number of ions
        if(n_neut > (n_spec - 1)) then
            err_message = "More neutrals than ions found!"
            err_info = error_info_t("Number of neutrals, ions was:", &
                                    [n_neut, n_spec - 1])
            ierr = GENEX_ERR_PARAMETERS
            goto 20
        end if

        ! Check that the corresponding ions of the neutrals species
        ! are indeed declared as plasma species
        do n = 1, n_spec
            plasma_species_names(n) = get_name(n)
        end do
        do o = 1, n_neut
            mapped_ions_names(o) = get_neut_mapped_ion_name(o)
            if(.not. any(mapped_ions_names(o) == plasma_species_names)) then
                err_message = "Neutrals that do not correspond to ions found!"
                err_info = error_info_t("Illegal neutral was:"&
                                        &//mapped_ions_names(o))
                ierr = GENEX_ERR_PARAMETERS
                goto 20
            end if
        end do

        ! Check only one ion per neutral
        do o = 1, n_neut
            n_ion_per_neut = 0
            do n = 1, n_spec
                if(plasma_species_names(n) == mapped_ions_names(o)) then
                    n_ion_per_neut = n_ion_per_neut + 1
                end if
            end do
            if(n_ion_per_neut > 1) then
                err_message = "Multiple neutrals for a single ion found!"
                err_info = error_info_t("Ion name and number of neutrals was:"&
                                        &//mapped_ions_names(o), &
                                        [n_ion_per_neut])
                ierr = GENEX_ERR_PARAMETERS
                goto 20
            end if
        end do
        deallocate(plasma_species_names, mapped_ions_names)

20      if(ierr /= GENEX_SUCCESS) then
            call handle_error(err_message, ierr, __LINE__, __FILE__, &
                              additional_info=err_info)
        end if

    end subroutine

    module subroutine initialize_diagnostics(this)
        class(timestep_t), intent(inout) :: this
        character(len=:), allocatable :: part_dir

        part_dir = create_part(this%dcomm_handler, trim(this%diag_dir))
        if(part_dir == "") then
            call handle_error(&
                    "Number of output parts exhausted!", &
                    GENEX_ERR_TIMESTEP, __LINE__, __FILE__)
        end if

        allocate(this%mom_0d_file)
        call this%mom_0d_file%initialize(this%dcomm_handler, &
            part_dir // "mom_0d.nc")
        allocate(this%mom_2d_file)
        call this%mom_2d_file%initialize(this%dcomm_handler, &
            part_dir // "mom_2d.nc", this%mesh%size_RZ(), &
            this%mesh%size_phi(), this%mesh%size_sp())
        if(get_diagnose_tpc()) then
            if(get_use_vspectral()) then
                call handle_error("TPC diagnostics are not supported &
                                  &with vspec!", &
                                  GENEX_ERR_TIMESTEP, __LINE__, __FILE__)
            else if(get_use_gpu_offload()) then
                call handle_error("TPC diagnostics do not work in GPU!", &
                                  GENEX_ERR_TIMESTEP, __LINE__, __FILE__)
            end if

            allocate(this%mom_2d_tpc_file)
            call this%mom_2d_tpc_file%initialize(this%dcomm_handler, &
                       part_dir // "mom_2d_tpc.nc", this%mesh%size_RZ(), &
                       this%mesh%size_phi(), this%mesh%size_sp())

        end if
        if(this%apply_neutrals) then
            allocate(this%neutrals_file)
            call this%neutrals_file%initialize(this%dcomm_handler, &
                part_dir // "neutrals.nc", this%mesh%size_RZ(), &
                this%mesh%size_phi(), get_n_points_neut())
            allocate(this%checkpoint_neut_file)
            call this%checkpoint_neut_file%initialize(this%dcomm_handler, &
                part_dir // "checkpoint_neut.nc", this%mesh%size_RZ(), &
                this%mesh%size_phi(), 1, this%mesh%size_sp())
        end if
        allocate(this%em_fields_file)
        call this%em_fields_file%initialize(this%dcomm_handler, &
            part_dir // "em_fields.nc", this%mesh%size_RZ(), &
            this%mesh%size_phi())
        allocate(this%checkpoint_file)
        call this%checkpoint_file%initialize(this%dcomm_handler, &
            part_dir // "checkpoint.nc", this%mesh%size_RZ(), &
            this%mesh%size_phi(), this%mesh%size_vp(), this%mesh%size_mu(), &
            this%mesh%size_sp())
    end subroutine

    module subroutine reinitialize_diagnostics(this)
        class(timestep_t), target, intent(inout) :: this

        ! Finalize the diagnostics in the old part directory
        call this%finalize_diagnostics()
        ! Create a new part directory and reinitialize the diagnostics in the
        ! new part directory
        call this%initialize_diagnostics()
    end subroutine

    module subroutine finalize_diagnostics(this)
        class(timestep_t), intent(inout) :: this

        deallocate(this%mom_0d_file, this%mom_2d_file, this%em_fields_file, &
                   this%checkpoint_file)
        if(get_diagnose_tpc()) then
            deallocate(this%mom_2d_tpc_file)
        end if
        if(this%apply_neutrals) then
            deallocate(this%neutrals_file, this%checkpoint_neut_file)
        end if
    end subroutine

    module subroutine initialize_operators(this)
        class(timestep_t), target, intent(inout) :: this

        character(len=:), allocatable :: neutrals_evolve_type, &
                                         neutrals_coupling_type
        real(kind=GP) :: n_norm, T_norm, k_norm

        if(this%dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) "   Initialize operators"
        end if

        ! Initialize MMS operators
        allocate(op_mms_source_vlasov_eq_cpu_t :: &
                this%op_mms_source_vlasov_eq)
        allocate(op_mms_source_maxwells_eq_cpu_t :: &
                this%op_mms_source_maxwells_eq)
        allocate(op_mms_source_ohms_law_cpu_t :: &
                 this%op_mms_source_ohms_law)
        allocate(op_mms_solution_vlasov_eq_cpu_t :: &
                 this%op_mms_solution_vlasov_eq)
        allocate(op_mms_solution_maxwells_eq_cpu_t :: &
                 this%op_mms_solution_maxwells_eq)
        allocate(op_mms_solution_ohms_law_cpu_t :: &
                 this%op_mms_solution_ohms_law)
        allocate(op_mms_error_cpu_t :: this%op_mms_error)

        call this%op_mms_source_vlasov_eq%initialize(this%mesh, this%bsg_op)
        call this%op_mms_source_maxwells_eq%initialize(this%mesh, this%bsg_op)
        call this%op_mms_source_ohms_law%initialize(this%mesh, this%bsg_op)
        call this%op_mms_solution_vlasov_eq%initialize(this%dcomm_handler, &
                                                       this%mesh)
        call this%op_mms_solution_maxwells_eq%initialize(this%mesh)
        call this%op_mms_solution_ohms_law%initialize(this%mesh)
        call this%op_mms_error%initialize(this%dcomm_handler, this%mesh, &
                                          this%bsg_op)

        call initialize_neutrals_operators()

        if(get_use_vspectral()) then
            call this%initialize_operators_vspec_cpu()
        else
            if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
                call this%initialize_operators_grid_gpu()
#endif
            else
                call this%initialize_operators_grid_cpu()
            end if
        end if

        ! Initialize field solve operators
        allocate(this%op_solve_qn_eq)
        allocate(this%op_solve_amps_law)
        allocate(this%op_solve_ohms_law)

        call this%op_solve_qn_eq%initialize(this%dcomm_handler, this%mesh)
        call this%op_solve_amps_law%initialize(this%dcomm_handler, &
                                               this%mesh)
        call this%op_solve_ohms_law%initialize(this%dcomm_handler, &
                                               this%mesh)

        if(get_with_bpar()) then
            allocate(this%op_solve_bpar_eq)
            call this%op_solve_bpar_eq%initialize(this%dcomm_handler, &
                                                  this%mesh)
        end if
    contains

        subroutine initialize_neutrals_operators

            neutrals_evolve_type = get_neut_evolve_type()
            select case(neutrals_evolve_type)
                case("none")

                case("dummy")
                    allocate(op_neut_evolve_dummy_t :: this%op_neut_evolve)
                    call this%op_neut_evolve%initialize(this%dcomm_handler, &
                                                        this%mesh)
                case("1mom_2cfd")
                    allocate(op_neut_evolve_1mom_2cfd_t :: this%op_neut_evolve)
                    call this%op_neut_evolve%initialize(this%dcomm_handler, &
                                                        this%mesh)
                case("1mom_4cfd")
                    allocate(op_neut_evolve_1mom_4cfd_t :: this%op_neut_evolve)
                    call this%op_neut_evolve%initialize(this%dcomm_handler, &
                                                        this%mesh)
                case("1mom_diff")
                    allocate(op_neut_evolve_1mom_diff_t :: this%op_neut_evolve)
                    call this%op_neut_evolve%initialize(this%dcomm_handler, &
                                                        this%mesh)
                case default
                    call handle_error(&
                        "Selected neutrals evolution type is not supported!", &
                        GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                        additional_info=&
                            error_info_t("Neutrals evolution type was: "&
                                         &//neutrals_evolve_type))
            end select

            neutrals_coupling_type = get_neut_coupling_type()
            select case(neutrals_coupling_type)
                case("none")

                case("dummy")
                    allocate(op_neut_mom_cpu_t :: this%op_neut_mom)
                    call this%op_neut_mom%initialize(this%dcomm_handler, &
                                                     this%mesh)
                    allocate(op_neut_coupling_np_none_t :: &
                                                       this%op_neut_coupling_np)
                    call this%op_neut_coupling_np%initialize(&
                                                        this%dcomm_handler, &
                                                        this%mesh)
                    if(neutrals_evolve_type == "1mom_2cfd" .or. &
                       neutrals_evolve_type == "1mom_4cfd" .or. &
                       neutrals_evolve_type == "1mom_diff") then
                        ! In the case of the evolution operator 1mom, the ion
                        ! temperature needs to be imported through the source.
                        ! op_neut_coupling_pn_temp is "dummy" with
                        ! respect to the density coupling but passes the ion
                        ! temperature through the temperature source term.
                        allocate(op_neut_coupling_pn_temp_t :: &
                                                      this%op_neut_coupling_pn)
                        call this%op_neut_coupling_pn%initialize( &
                                                           this%dcomm_handler, &
                                                           this%mesh)
                    else
                        allocate(op_neut_coupling_pn_none_t :: &
                                                       this%op_neut_coupling_pn)
                        call this%op_neut_coupling_pn%initialize( &
                                                           this%dcomm_handler, &
                                                           this%mesh)
                    end if

                case("krook")
                    ! NOTE: The coupling is performed on the density only (1mom)
                    allocate(op_neut_mom_cpu_t :: this%op_neut_mom)
                    call this%op_neut_mom%initialize(this%dcomm_handler, &
                                                     this%mesh)

                    ! NOTE: The conservative Krook coupling operators (Wersal
                    !       inspired) work for ONLY ONE neutrals species.
                    if(get_n_points_neut() == 1) then
                        allocate(op_neut_coupling_np_krook_t :: &
                                                      this%op_neut_coupling_np)
                        allocate(op_neut_coupling_pn_krook_t :: &
                                                     this%op_neut_coupling_pn)
                        ! REAX density in cm^{-3}
                        n_norm = get_n_ref() * 1.0e13_GP
                        ! REAX temperature in eV
                        T_norm = get_T_ref() * 1.0e3_GP
                        ! REAX reaction rate in cm^3 s^{-1}
                        k_norm = get_rate_ref() * 1.0e-13_GP
                        call reax_initialize_rate_norm(n_norm, T_norm, k_norm)
                    else
                        call handle_error(&
                            "No Krook coupling operators available for more " &
                            // "than one neutrals species!", &
                            GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                            additional_info=&
                                error_info_t("Neutrals coupling type was: " &
                                             // neutrals_coupling_type &
                                             // " with " &
                                             // string(get_n_points_neut()) &
                                             // " neutrals species"))
                    end if
                    call this%op_neut_coupling_np%initialize(&
                                                         this%dcomm_handler, &
                                                         this%mesh)
                    call this%op_neut_coupling_pn%initialize(&
                                                     this%dcomm_handler, &
                                                     this%mesh)
                case default
                    call handle_error(&
                         "Selected neutrals coupling type is not supported!", &
                         GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                         additional_info=&
                             error_info_t("Neutrals coupling type was: "&
                                             &//neutrals_coupling_type))
            end select

            this%apply_neutrals = get_use_neutrals()
        end subroutine
    end subroutine

    module subroutine initialize_state_vector(this)
        class(timestep_t), target, intent(inout) :: this
        character(len=:), allocatable :: part_dir

        if(this%dcomm_handler%is_master()) then
            write(logger_get_info_channel(), *) "   Initialize state vector"
        end if

        call initialize_set_initial_operators()

        allocate(this%state)
        call this%state%initialize(this%dcomm_handler)

        call this%state%apply(this%op_set_initial_condition)
        if(this%apply_neutrals) then
            call this%state%apply(this%op_neut_set_initial)
        end if

        if(this%start_from_checkpoint) then
            if(this%dcomm_handler%is_master()) then
                write(logger_get_info_channel(), *) &
                    "   Reading initial condition from checkpoint file"
            end if
            part_dir = inquire_part(this%dcomm_handler, trim(this%diag_dir))
            if(part_dir == "") then
                if(this%dcomm_handler%is_master()) then
                    call handle_error(&
                            "Start_from_checkpoint activated but no checkpoint &
                            &found! Starting the simulation from the initial &
                            &state.", GENEX_WRN_FILE_NOT_EXIST, &
                            __LINE__, __FILE__)
                end if
                this%t = 0.0_GP
                this%start_from_checkpoint = .false.
            else
                allocate(this%checkpoint_file)
                call this%checkpoint_file%initialize(&
                    this%dcomm_handler, part_dir // "checkpoint.nc", &
                    this%mesh%size_RZ(), this%mesh%size_phi(), &
                    this%mesh%size_vp(), this%mesh%size_mu(), &
                    this%mesh%size_sp(), from_existing=.true.)
                call this%state%read_checkpoint(this%checkpoint_file, this%t)
                deallocate(this%checkpoint_file)

                if(this%apply_neutrals) then
                    allocate(this%checkpoint_neut_file)
                    call this%checkpoint_neut_file%initialize(&
                        this%dcomm_handler, part_dir // "checkpoint_neut.nc", &
                        n_points_RZ=this%mesh%size_RZ(), &
                        n_points_phi=this%mesh%size_phi(), &
                        n_points_mom=1, &
                        n_points_sp=get_n_points_neut(), &
                        from_existing=.true.)
                    call this%state%read_checkpoint_neut(&
                        this%checkpoint_neut_file, this%t)
                    deallocate(this%checkpoint_neut_file)
                end if

                if(this%dcomm_handler%is_master()) then
                    write(logger_get_info_channel(), *) &
                        "   Obtained simulation time is t = ", this%t
                end if
            end if
        else
            this%t = 0.0_GP
        end if

        ! NOTE: Move these update device inside the checkpoint if-guard if
        !       op_set_initial_condition and op_neut_set_initial become
        !       GPU-supported
        call this%state%update_device_dist_func()

    contains

        subroutine initialize_set_initial_operators
            if(this%run_mms) then
                allocate(op_set_initial_condition_mms_cpu_t :: &
                         this%op_set_initial_condition)
            else
                if(get_use_vspectral()) then
                    allocate(op_set_initial_condition_vspec_cpu_t :: &
                             this%op_set_initial_condition)
                else
                    allocate(op_set_initial_condition_cpu_t :: &
                             this%op_set_initial_condition)
                end if
            end if

            select type(op => this%op_set_initial_condition)
                type is(op_set_initial_condition_mms_cpu_t)
                    call op%initialize(this%mesh)
                type is(op_set_initial_condition_cpu_t)
                    call op%initialize(this%dcomm_handler, &
                                       this%mesh, &
                                       this%dist_initial_container, &
                                       this%bsg_op)
                type is(op_set_initial_condition_vspec_cpu_t)
                    call op%initialize(this%dcomm_handler, &
                                       this%mesh, &
                                       this%dist_initial_container)
            end select

            if(this%apply_neutrals) then
                allocate(op_neut_set_initial_cpu_t :: this%op_neut_set_initial)
                call this%op_neut_set_initial%initialize(this%dcomm_handler, &
                                                         this%mesh)
            end if
        end subroutine
    end subroutine

end submodule
