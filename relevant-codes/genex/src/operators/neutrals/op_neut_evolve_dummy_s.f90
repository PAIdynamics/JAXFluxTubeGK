submodule(op_neut_evolve_m) op_neut_evolve_dummy_s
    !! Contains implementation of op_neut_evolve_dummy_t which represents
    !! a "dummy" (minimal working but not physical) neutrals evolution operator.

    implicit none

contains

    module subroutine initialize_evolve_dummy(this, dcomm_handler, mesh)
        class(op_neut_evolve_dummy_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%dcomm_handler => dcomm_handler
        this%mesh          => mesh

    end subroutine

    module subroutine apply_evolve_dummy(this, da_n_sources, &
                                               da_n_in, da_n_out)
        !! This subroutine evolves the neutrals species according to the
        !! dummy model, ie no modification
        class(op_neut_evolve_dummy_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_n_sources
        class(data_array_4d_t), intent(in) :: da_n_in
        class(data_array_4d_t), intent(inout) :: da_n_out

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: n_sources, &
                                                                  n_in, n_out
        real(kind=GP), dimension(:,:), contiguous, pointer :: is_compute
        integer, dimension(4) :: lb_stripped, ub_stripped
        integer :: o, k, i

        n_sources => da_n_sources%get_readonly_pointer()
        n_in => da_n_in%get_readonly_pointer()
        n_out => da_n_out%get_pointer()

        lb_stripped = da_n_in%get_lbound_stripped()
        ub_stripped = da_n_in%get_ubound_stripped()

        is_compute => this%mesh%get_is_compute_pointer()

        call this%perf_counter%start_measurement()
        ! Nothing done here, only add src term (coupling might not be dummy)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped, is_compute) &
        !$omp shared(this, n_out, n_sources) &
        !$omp private(i, k, o)
        do o = lb_stripped(4), ub_stripped(4)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)
            n_out(i, k, 1, o) = n_sources(i, k, 1, o)
            n_out(i, k, 1, o) = n_out(i, k, 1, o) * is_compute(i, k)
        end do
        !$omp end do nowait
        end do
        end do
        !$omp end parallel
        call this%perf_counter%end_measurement()

    end subroutine

end submodule
