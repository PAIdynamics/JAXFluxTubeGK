submodule(op_solve_bpar_eq_m) op_solve_bpar_eq_s
    use logger_m, only: logger_get_debug_channel
    use profiler_m, only: profiler_start, profiler_stop
#ifdef ENABLE_GPU
    use op_copy_m, only: op_copy_gpu_t
#else
    use op_copy_m, only: op_copy_cpu_t
#endif

    implicit none

contains

    module subroutine initialize(this, dcomm_handler, mesh)
        class(op_solve_bpar_eq_t), intent(inout) :: this
        type(dcomm_handler_t), target, intent(inout) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
    end subroutine

    module subroutine apply(this, da_b_bpar_eq, da_B_par)
        class(op_solve_bpar_eq_t), target, intent(inout) :: this
        class(data_array_2d_t), intent(in) :: da_b_bpar_eq
        class(data_array_2d_t), intent(inout) :: da_B_par

        integer :: ierr
        real(kind=GP), contiguous, pointer, dimension(:,:) :: b_bpar_eq, B_par
#ifdef ENABLE_GPU
        type(op_copy_gpu_t) :: op_copy
#else
        type(op_copy_cpu_t) :: op_copy
#endif

        call op_copy%initialize()
        b_bpar_eq => da_b_bpar_eq%get_readonly_pointer()
        B_par => da_B_par%get_pointer_stripped()

        call profiler_start("copy", ierr)
        call op_copy%apply(B_par, b_bpar_eq)
        call profiler_stop("copy", ierr)
    end subroutine

end submodule
