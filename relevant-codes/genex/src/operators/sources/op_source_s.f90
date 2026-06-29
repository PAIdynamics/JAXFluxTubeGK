submodule (op_source_m) op_source_s
    use genex_error_handling_m, only: handle_error_notimp
     use, intrinsic :: iso_fortran_env
     use params_source_m, only: params_source_loc_t, &
                                n_source_loc_supported
     use op_set_uniform_m, only: op_set_uniform_cpu_t
     implicit none

contains
    module subroutine initialize_source_norm_fac(this, s, n)
        class(op_source_loc_t), intent(inout) :: this
        integer, intent(in) :: s
        integer, intent(in) :: n

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

end submodule
