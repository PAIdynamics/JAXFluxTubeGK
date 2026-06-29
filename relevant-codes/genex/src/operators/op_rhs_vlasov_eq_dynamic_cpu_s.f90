submodule (op_rhs_vlasov_eq_dynamic_m) op_rhs_vlasov_eq_dynamic_cpu_s
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP_EPS
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use bsg_types_m, only: unpack_bflag

    implicit none

contains

    module subroutine initialize_cpu(this, mesh, bsg_op)
        class(op_rhs_vlasov_eq_dynamic_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(bsg_operators_t), target, intent(inout) :: bsg_op
        integer :: n, n_points_sp

        this%n_flops = 13   ! We estimate division with 4
        this%n_ls    = 8

        this%mesh => mesh
        this%bsg_op => bsg_op

        n_points_sp = mesh%size_sp()
        allocate(this%prefac_norm(n_points_sp))
        do n = 1, n_points_sp
            this%prefac_norm(n) = get_charge(n) &
                                / sqrt(2.0_GP * get_mass(n) &
                                              * get_temp_scaling(n))
        enddo
    end subroutine

    module subroutine apply_cpu(this, da_f_in, da_E_par_in, da_f_out)
        class(op_rhs_vlasov_eq_dynamic_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_E_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        integer :: i, k, l, m, n
        integer, dimension(5) :: lb_stripped, ub_stripped
        real(kind=GP) :: dfdvp
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_in, f_out
        real(kind=GP), contiguous, pointer, dimension(:,:) :: E_par_in
        real(kind=GP), contiguous, pointer, dimension(:,:) :: is_compute
        integer, contiguous, pointer, dimension(:) :: bsg_flags
        real(kind=GP), contiguous, dimension(:), pointer :: prefac_cd_vp_bsg
        integer :: nb_point

        lb_stripped = da_f_in%get_lbound_stripped()
        ub_stripped = da_f_in%get_ubound_stripped()

        f_in => da_f_in%get_readonly_pointer()
        E_par_in => da_E_par_in%get_readonly_pointer()
        f_out => da_f_out%get_pointer()

        this%n_iterations = da_f_in%get_size_stripped()
        call this%perf_counter%start_measurement()

        is_compute => this%mesh%get_is_compute_pointer()
        bsg_flags => this%mesh%get_bsg_flags_pointer()
        prefac_cd_vp_bsg => this%bsg_op%get_prefac_vp_pointer()

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, is_compute, f_in, E_par_in, f_out, &
        !$omp        bsg_flags, prefac_cd_vp_bsg) &
        !$omp private(i, k, l, m, n, dfdvp, nb_point)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do simd schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            nb_point = unpack_bflag(bsg_flags(i))
            ! Calculate vp derivatives
            dfdvp = prefac_cd_vp_bsg(nb_point) * &
                    (- 1.0_GP / 12.0_GP * f_in(i,k,l + 2,m,n) &
                     + 2.0_GP / 3.0_GP  * f_in(i,k,l + 1,m,n) &
                     - 2.0_GP / 3.0_GP  * f_in(i,k,l - 1,m,n) &
                     + 1.0_GP / 12.0_GP * f_in(i,k,l - 2,m,n))
            ! Calculate the electromagnetic induction
            f_out(i, k, l, m, n) = f_out(i, k, l, m, n) &
                                 + this%prefac_norm(n) * E_par_in(i, k) &
                                 * dfdvp * is_compute(i, k)
        enddo
        !$omp end do simd nowait
        enddo
        enddo
        enddo
        enddo
        !$omp end parallel

        call this%perf_counter%end_measurement()
    end subroutine

end submodule
