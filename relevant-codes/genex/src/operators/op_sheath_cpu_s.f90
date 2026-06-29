submodule (op_sheath_m) op_sheath_s
    use mpi
    use genex_fortran_env_m, only : MPI_GP
    use math_m, only : PI, is_abs_lt, is_abs_le
    use params_species_m, only : get_mass, get_charge
    use params_normalization_m, only : get_rho_ref, get_L_ref
    use csralg_m, only: csrmv
    ! from PARALLAX
    use equilibrium_factory_m, only : SLAB
    implicit none

contains

    module subroutine initialize_cpu(this, dcomm_handler, mesh)
        class(op_sheath_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh
        integer :: n, n_points_sp

        this%mesh => mesh
        this%dcomm_handler => dcomm_handler
    end subroutine

    module subroutine apply_cpu(this, lb, ub, lb_stripped, ub_stripped, &
                                dist_func, is_electron_sheath, cutoff_index, &
                                cutoff_index_fraction, sheath_pot)
        class(op_sheath_cpu_t), intent(inout) :: this
        integer, dimension(5), intent(in) :: lb, ub
        integer, dimension(5), intent(in) :: lb_stripped, ub_stripped
        real(kind = GP), intent(in) :: dist_func(lb(1) : ub(1), &
                                                 lb(2) : ub(2), &
                                                 lb(3) : ub(3), &
                                                 lb(4) : ub(4), &
                                                 lb(5) : ub(5) )
        logical, intent(out) :: is_electron_sheath( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
        integer, intent(out) :: cutoff_index( &
            lb(1)          : ub(1), &
            lb_stripped(2) : ub_stripped(2))
        real(kind = GP), intent(out) :: cutoff_index_fraction( &
            lb(1)          : ub(1), &
            lb_stripped(2) : ub_stripped(2))
        real(kind = GP), intent(out) :: sheath_pot( &
            lb(1)          : ub(1), &
            lb_stripped(2) : ub_stripped(2))
        integer :: i, k

        !$omp parallel firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(this, sheath_pot, cutoff_index, cutoff_index_fraction, &
        !$omp        is_electron_sheath) &
        !$omp private(i, k)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            is_electron_sheath(i, k) = .true.
            sheath_pot(i, k) = 0.0_GP
            cutoff_index_fraction(i, k) = 1.0_GP
            cutoff_index(i, k) = 0
        enddo
        !$omp end do
        enddo
        !$omp end parallel
    end subroutine
end submodule
