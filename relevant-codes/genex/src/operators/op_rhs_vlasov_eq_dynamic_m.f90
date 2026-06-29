module op_rhs_vlasov_eq_dynamic_m
    !! Contains an operator to calculate time dependent RHS parts
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use mesh_5d_m, only: mesh_5d_t
    use bsg_operators_m, only: bsg_operators_t

    implicit none

    type, public, abstract, extends(op_base_t) :: &
                                                op_rhs_vlasov_eq_dynamic_base_t
        !! Base operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time derivatives
        class(mesh_5d_t), pointer :: mesh
        !! Pointer to the mesh
        class(bsg_operators_t), pointer :: bsg_op
        !! Pointer to the BSG operators
    contains
        procedure(apply_interface), deferred :: apply
    end type

    interface
        subroutine apply_interface(this, da_f_in, da_E_par_in, da_f_out)
            !! Applies the operator to the given input values and adds the
            !! result to f_out
            import op_rhs_vlasov_eq_dynamic_base_t, data_array_2d_t, &
                                                          data_array_5d_t
            class(op_rhs_vlasov_eq_dynamic_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_2d_t), intent(in) :: da_E_par_in
            !! Input parallel electric field
            class(data_array_5d_t), intent(inout) :: da_f_out
            !! Output distribution function
        end subroutine
    end interface

    type, public, abstract, extends(op_rhs_vlasov_eq_dynamic_base_t) :: &
                                                      op_rhs_vlasov_eq_dynamic_t
        !! Operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time derivatives
    contains
        procedure(initialize_interface), deferred :: initialize
    end type

    interface
        module subroutine initialize_interface(this, mesh, bsg_op)
            !! Initializes the op_rhs_vlasov_eq_dynamic_t type
            class(op_rhs_vlasov_eq_dynamic_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! BSG operators
        end subroutine
    end interface

    type, public, extends(op_rhs_vlasov_eq_dynamic_t) :: &
                                                  op_rhs_vlasov_eq_dynamic_cpu_t
        !! Operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time derivatives
        !! on the CPU
        real(kind=GP), allocatable, dimension(:) :: prefac_norm
        !! Normalization prefactor of the induction term
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, mesh, bsg_op)
            class(op_rhs_vlasov_eq_dynamic_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, da_f_in, da_E_par_in, da_f_out)
            class(op_rhs_vlasov_eq_dynamic_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_E_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, abstract, extends(op_rhs_vlasov_eq_dynamic_base_t) :: &
                                                op_rhs_vlasov_eq_dynamic_vspec_t
        !! Operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time
        !! derivatives using spectral method.
    contains
        procedure(initialize_interface_vspec), deferred :: initialize
    end type

    interface
        module subroutine initialize_interface_vspec(this, mesh)
            !! Initializes the op_rhs_vlasov_eq_dynamic_vspec_t type
            class(op_rhs_vlasov_eq_dynamic_vspec_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_rhs_vlasov_eq_dynamic_vspec_t) :: &
                                            op_rhs_vlasov_eq_dynamic_vspec_cpu_t
        !! Operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time
        !! derivatives using spectral method on CPU.
        real(kind=GP), allocatable, dimension(:) :: prefac_norm
        !! First normalization prefactor
        real(kind=GP), pointer, contiguous, dimension(:) :: vp
        !! Vp grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vp
        !! sqrt(vp) grid
        real(kind=GP), private, allocatable, dimension(:) :: temp_scalings
        !! Temperature scaling
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, mesh)
            class(op_rhs_vlasov_eq_dynamic_vspec_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_vspec_cpu(this, da_f_in, da_E_par_in, da_f_out)
            class(op_rhs_vlasov_eq_dynamic_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_E_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_rhs_vlasov_eq_dynamic_t) :: &
        op_rhs_vlasov_eq_dynamic_gpu_t
        !! Operator to calculate the terms of the right-hand side of the
        !! gyrokinetic Vlasov equation that include explicit time derivatives
        !! on the GPU
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_rhs_vlasov_eq_dynamic_gpu_t C++ class instance
        !! pointer
    contains
        procedure :: initialize => initialize_gpu
        procedure :: apply => apply_gpu
        final     :: finalize_gpu
    end type

    interface
        module subroutine initialize_gpu(this, mesh, bsg_op)
            class(op_rhs_vlasov_eq_dynamic_gpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_gpu(this, da_f_in, da_E_par_in, da_f_out)
            class(op_rhs_vlasov_eq_dynamic_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_E_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine finalize_gpu(this)
            type(op_rhs_vlasov_eq_dynamic_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif
end module
