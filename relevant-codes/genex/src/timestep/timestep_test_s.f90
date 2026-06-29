submodule (timestep_m) timestep_test_s
    !! To test the time step types in a unit test we replace the core routines
    !! in timestep by test subroutines that will simulate the workload of the
    !! timestep by test functions that can be analytically integrated. This
    !! result can then be used to test the timestep types.
    !!
    !! Form of the analytical equation:
    !! dx/dt       = static(t) + dynamic(t) + coll(t) + neutrals(t)
    !! static(t)   = maxw(t) * x
    !! dynamic(t)  = ohm(t)  * x
    !! coll(t)     = sin(t)  * x
    !! neutrals(t) = cos(t)  * x
    !! maxw(t)     = -4
    !! ohm(t)      = -2t
    !! x(t=0)      = 1
    !!
    !! Analytical solution:
    !! x(t)       = exp(1 - 4t - t**2 - cos(t) + sin(t))
    !!
    !! We choose this form to make use of all subroutines in the timestep
    !! to test the correct interplay within the time integration.

    ! NOTE: Because the test overwrites the private calc and evolve procedures,
    !       the test versions need to be located in the same module. Thus, we
    !       chose the version using procedure pointers to redirect the core
    !       procedures.

    implicit none

contains

    module subroutine setup_test_mode(this, run_test)
        class(timestep_t), intent(inout) :: this
        logical, intent(in) :: run_test
        if(run_test) then
            this%calc_rhs_static   => calc_rhs_static_test
            this%calc_rhs_dynamic  => calc_rhs_dynamic_test
            this%calc_collisions   => calc_collisions_test
            this%calc_neutrals     => calc_neutrals_test
            this%solve_maxwells_eq => solve_maxwells_eq_test
            this%solve_ohms_law    => solve_ohms_law_test
            this%apply_bnd_cond    => apply_bnd_cond_test

            this%apply_plasma     = .true.
            this%apply_collisions = .true.
            this%apply_neutrals   = .true.
        endif

        this%run_test = run_test
    end subroutine

    subroutine calc_rhs_static_test(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        call s_out%set_uniform(0.0_GP)
        call s_out%add(this%test_result_maxw, s_in)
    end subroutine

    subroutine calc_rhs_dynamic_test(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        call s_out%add(this%test_result_ohm, s_in)
    end subroutine

    subroutine calc_collisions_test(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out
        call s_out%add(sin(this%t), s_in)
    end subroutine

    subroutine calc_neutrals_test(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out
        call s_out%add(cos(this%t), s_in)
    end subroutine

    subroutine solve_maxwells_eq_test(this, s_inout, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_inout
        integer, intent(out) :: ierr

        ierr = GENEX_SUCCESS
        this%test_result_maxw = -4.0_GP
    end subroutine

    subroutine solve_ohms_law_test(this, dsdt_in, s_inout, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: dsdt_in
        type(state_vector_t), target, intent(inout) :: s_inout
        integer, intent(out) :: ierr

        ierr = GENEX_SUCCESS
        this%test_result_ohm = -2.0_GP * this%t
    end subroutine

    subroutine apply_bnd_cond_test(this, s_inout)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_inout
    end subroutine

end submodule
