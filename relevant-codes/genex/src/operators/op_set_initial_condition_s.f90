submodule (op_set_initial_condition_m) op_set_initial_condition_s

    implicit none

contains

    module subroutine set_noise_perturbation(this, da_f_inout)
        class(op_set_initial_condition_base_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        integer :: i, k, l, m, n, seed_size
        integer, dimension(5) :: lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), pointer, dimension(:,:) :: is_compute
        integer, allocatable, dimension(:) :: seed
        real(kind=GP) :: noise_level, rand

        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout => da_f_inout%get_pointer()
        is_compute => this%mesh%get_is_compute_pointer()

        noise_level = 1e-3_GP

        ! initialize random number generator
        call random_seed(size = seed_size)
        allocate(seed(seed_size))
        do i = 1, seed_size
           seed(i) = i + lb_stripped(2) * 10000
        end do
        call random_seed(put = seed)

        ! NOTE: the random number generation is not thread safe
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        do i = lb_stripped(1), ub_stripped(1)
            call random_number(rand)
            f_inout(i, k, l, m, n) = f_inout(i, k, l, m, n) &
                                   * (1.0_GP + noise_level * rand &
                                             * is_compute(i, k))
        enddo
        enddo
        enddo
        enddo
        enddo
    end subroutine

end submodule
