submodule (op_neut_set_initial_m) op_neut_set_initial_cpu_s
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_fortran_env_m, only: GP
    use math_m, only: PI
    use data_array_m, only: data_array_4d_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use params_neutrals_m, only: get_n_points_neut
    use params_neutrals_init_m, only: get_neut_profile_type_dens, &
                                      get_neut_initial_perturbation_dens
    use params_neutrals_set_blob_m, only: params_neut_blob_t, &
                                          get_neut_params_blob_dens
    use params_profile_uniform_m, only: params_profile_uniform_t, &
                                        get_params_profile_uniform_dens
    use params_profile_plateau_m, only: params_profile_plateau_t, &
                                        get_params_profile_plateau_dens
    use profiles_m, only: profile_uniform_cpu_t, profile_plateau_sine_cpu_t

    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh)
        class(op_neut_set_initial_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(in) :: mesh

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
    end subroutine

    module subroutine apply_uniform_cpu(this, neut_state, neut_mom, neut_spec)
        class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
        class(data_array_4d_t), intent(inout) :: neut_state
        integer, intent(in) :: neut_mom
        integer, intent(in) :: neut_spec

        integer :: i, k
        integer, dimension(4) :: lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: neut
        real(kind=GP), contiguous, pointer, dimension(:,:) :: rho, theta
        real(kind=GP), contiguous, pointer, dimension(:) :: phi
        type(params_profile_uniform_t) :: params
        type(profile_uniform_cpu_t) :: profile

        lb_stripped = neut_state%get_lbound_stripped()
        ub_stripped = neut_state%get_ubound_stripped()

        neut => neut_state%get_pointer()

        rho   => this%mesh%get_rho_pointer()
        theta => this%mesh%get_theta_pointer()
        phi   => this%mesh%get_phi_pointer()

        if (neut_mom == 1) params = get_params_profile_uniform_dens(neut_spec)

        call profile%initialize(params)
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, neut, neut_mom, neut_spec, &
        !$omp        profile, rho, theta, phi) &
        !$omp private(i, k)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            neut(i, k, neut_mom, neut_spec) = profile%eval(rho(i, k), &
                                                           theta(i, k), &
                                                           phi(k))
        end do
        !$omp end do nowait
        end do
        !$omp end parallel
    end subroutine

    module subroutine apply_plateau_sine_cpu(this, neut_state, &
                                             neut_mom, neut_spec)
        class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
        class(data_array_4d_t), intent(inout) :: neut_state
        integer, intent(in) :: neut_mom
        integer, intent(in) :: neut_spec

        integer, dimension(4) :: lb, ub, lb_stripped, ub_stripped
        integer :: i, k
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: neut
        real(kind=GP), contiguous, pointer, dimension(:,:) :: rho, theta
        real(kind=GP), contiguous, pointer, dimension(:) :: phi
        type(params_profile_plateau_t) :: params
        type(profile_plateau_sine_cpu_t) :: profile

        lb = neut_state%get_lbound()
        ub = neut_state%get_ubound()
        lb_stripped = neut_state%get_lbound_stripped()
        ub_stripped = neut_state%get_ubound_stripped()

        neut => neut_state%get_pointer()

        rho   => this%mesh%get_rho_pointer()
        theta => this%mesh%get_theta_pointer()
        phi   => this%mesh%get_phi_pointer()

        if (neut_mom==1) params = get_params_profile_plateau_dens(neut_spec)

        call profile%initialize(params)
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, neut, neut_mom, neut_spec, &
        !$omp        profile, rho, theta, phi) &
        !$omp private(i, k)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            neut(i, k, neut_mom, neut_spec) = profile%eval(rho(i, k), &
                                                           theta(i, k), &
                                                           phi(k))
        end do
        !$omp end do nowait
        end do
        !$omp end parallel
    end subroutine

    module subroutine set_blob_perturbation_cpu(this, neut_state, &
                                                neut_mom, neut_spec)
        class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
        class(data_array_4d_t), intent(inout) :: neut_state
        integer, intent(in) :: neut_mom
        integer, intent(in) :: neut_spec

        integer :: i, k
        integer, dimension(4) :: lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: neut
        real(kind=GP), contiguous, pointer, dimension(:,:) :: R, Z
        real(kind=GP), contiguous, pointer, dimension(:) :: phi
        real(kind=GP), allocatable, dimension(:) :: mean_R_trace, mean_Z_trace
        real(kind=GP) :: inv_sigma_R, inv_sigma_Z, inv_sigma_phi
        real(kind=GP) :: exp_R_term, exp_Z_term, exp_phi_term
        real(kind=GP) :: mean_R, mean_Z, mean_phi, &
                         sigma_R, sigma_Z, sigma_phi, &
                         strength
        type(params_neut_blob_t) :: params

        lb_stripped = neut_state%get_lbound_stripped()
        ub_stripped = neut_state%get_ubound_stripped()
        neut => neut_state%get_pointer()

        R   => this%mesh%get_R_pointer()
        Z   => this%mesh%get_Z_pointer()
        phi => this%mesh%get_phi_pointer()

        if (neut_mom==1) params = get_neut_params_blob_dens(neut_spec)

        mean_R = params%mean_R
        mean_Z = params%mean_Z
        mean_phi = params%mean_phi
        sigma_R = params%sigma_R
        sigma_Z = params%sigma_Z
        sigma_phi = params%sigma_phi
        strength = params%strength

        inv_sigma_R = 1.0_GP / sigma_R
        inv_sigma_Z = 1.0_GP / sigma_Z
        inv_sigma_phi = 1.0_GP / sigma_phi

        allocate(mean_R_trace(this%mesh%size_phi()))
        allocate(mean_Z_trace(this%mesh%size_phi()))

        ! Determine center of field-aligned blob on each plane
        ! NOTE: A neutral blob does not need to be field aligned but given that
        !       the plasma temperature is used as a proxy for the neutrals
        !       temperature, the initialization of the neutral blob is based on
        !       the one for the plasma (this way a plasma and a neutral blob can
        !       be initialised in the same location).

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, neut, neut_mom, neut_spec, &
        !$omp        mean_R_trace, mean_Z_trace, R, Z, phi, &
        !$omp        mean_R, mean_Z, mean_phi, &
        !$omp        sigma_R, sigma_Z, sigma_phi, strength, &
        !$omp        inv_sigma_R, inv_sigma_Z, inv_sigma_phi) &
        !$omp private(k, i, exp_R_term, exp_Z_term, exp_phi_term)
        !$omp do schedule(static)
        do k = 1, this%mesh%size_phi()
            call this%mesh%trace(x=mean_R, &
                                 y=mean_Z, &
                                 phi=mean_phi, &
                                 delta_phi=(phi(k) - mean_phi), &
                                 xp=mean_R_trace(k), &
                                 yp=mean_Z_trace(k))
        end do
        !$omp end do
        !$omp barrier
        ! !$omp do collapse(2) schedule(static)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            exp_R_term   = ((R(i, k) - mean_R_trace(k)) * inv_sigma_R)**2
            exp_Z_term   = ((Z(i, k) - mean_Z_trace(k)) * inv_sigma_Z)**2
            exp_phi_term = ((phi(k) - mean_phi) * inv_sigma_phi)**2

            neut(i, k, neut_mom, neut_spec) = neut(i, k, neut_mom, neut_spec) &
                                            + strength &
                                              * exp(-0.5 * (exp_R_term &
                                                           + exp_Z_term &
                                                           + exp_phi_term))
        end do
        !$omp end do
        end do
        !$omp end parallel

        deallocate(mean_R_trace, mean_Z_trace)
    end subroutine

    module subroutine set_noise_perturbation_cpu(this, neut_state, &
                                                 neut_mom, neut_spec)
        class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
        class(data_array_4d_t), intent(inout) :: neut_state
        integer, intent(in) :: neut_mom
        integer, intent(in) :: neut_spec

        integer :: i, k, seed_size
        integer, allocatable, dimension(:) :: seed
        integer, dimension(4) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: neut
        real(kind=GP), pointer, dimension(:,:) :: is_compute
        real(kind=GP) :: rand

        lb = neut_state%get_lbound()
        ub = neut_state%get_ubound()
        lb_stripped = neut_state%get_lbound_stripped()
        ub_stripped = neut_state%get_ubound_stripped()

        neut => neut_state%get_pointer()

        ! Initialize random number generator
        call random_seed(size=seed_size)
        allocate(seed(seed_size))
        do i = 1, seed_size
            seed(i) = i + lb_stripped(2) * 10000
        end do
        call random_seed(put=seed)

        ! NOTE: The random number generation is not thread safe
        do k = lb_stripped(2), ub_stripped(2)
        do i = lb_stripped(1), ub_stripped(1)
            call random_number(rand)
            neut(i, k, neut_mom, neut_spec) = neut(i, k, neut_mom, neut_spec) &
                                            * (1.0_GP + 1.0e-3_GP * rand &
                                               * is_compute(i, k))
        end do
        end do
    end subroutine

    module subroutine apply_cpu(this, neut_state)
        class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
        class(data_array_4d_t), intent(inout) :: neut_state

        integer :: neut_mom, o

        ! NOTE: Given the current evolution operator for the neutrals, only the
        !       neutral density needs to be initialized (the temperature of the
        !       neutrals is assumed to be proportional to the one of the plasma)
        do o = 1, get_n_points_neut()
            neut_mom = 1
            select case(get_neut_profile_type_dens(o))
                case ("uniform")
                    call this%apply_uniform_cpu(neut_state, neut_mom, o)
                case ("plateau_sine")
                    call this%apply_plateau_sine_cpu(neut_state, neut_mom, o)
                case default
                    call handle_error(&
                            "Selected initial profile is not supported!", &
                            GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                            additional_info=&
                                error_info_t("Profile was: "&
                                             &//get_neut_profile_type_dens(o)))
            end select

            select case(get_neut_initial_perturbation_dens(o))
                case ("noise")
                    call this%set_noise_perturbation_cpu(neut_state, &
                                                         neut_mom, o)
                case ("blob")
                    call this%set_blob_perturbation_cpu(neut_state, &
                                                        neut_mom, o)
                case ("none")
                    cycle
                case default
                    call handle_error(&
                            "Selected initial perturbation is not supported!", &
                            GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                            additional_info=&
                                error_info_t("Perturbation was: "&
                                    &//get_neut_initial_perturbation_dens(o)))
            end select
        end do
    end subroutine

end submodule
