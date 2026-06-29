module op_rhs_vlasov_eq_static_m
    !! Contains an operator to calculate time independent RHS parts
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP, CP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_operators_m, only: bsg_operators_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_rhs_vlasov_eq_static_base_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives
        class(mesh_5d_t), pointer :: mesh
        !! Pointer to the mesh
        class(bsg_operators_t), pointer :: bsg_op
        !! Pointer to the BSG operator
        real(kind=GP) :: prefac_arakawa
        !! Prefactor for Arakawa bracket
        real(kind=GP) :: prefac_cd_xz
        !! Prefactor for the centered finite difference in orthogonal direction
        real(kind=GP) :: prefac_hyp_xz
        !! Prefactor for the hyperdiffusion in orthogonal direction
        real(kind=GP) :: prefac_H2
        !! Prefactor for the second order Hamiltonian
        real(kind=GP), allocatable, dimension(:) :: prefac_Bpar
        !! Prefactor for the parallel magnetic fluctuations (on/off switch)
        !! with temperature scaling included
        real(kind=GP), allocatable, dimension(:) :: charges
        !! Species charges
        real(kind=GP), allocatable, dimension(:) :: masses
        !! Species masses
        real(kind=GP), allocatable, dimension(:) :: temp_scalings
        !! Species temperature scaling factors
        real(kind=GP), allocatable, dimension(:,:) :: prefac_buffer_zone
        !! Prefactor for the diffusion applied in the buffer zone
        real(kind=GP), allocatable, dimension(:,:) :: prefac_cd_y_mm
        !! Prefactor of the k - 2 contribution in the centered finite
        !! difference stencil of the parallel derivative
        real(kind=GP), allocatable, dimension(:,:) :: prefac_cd_y_m
        !! Prefactor of the k - 1 contribution in the centered finite
        !! difference stencil of the parallel derivative
        real(kind=GP), allocatable, dimension(:,:) :: prefac_cd_y_o
        !! Prefactor of the k contribution in the centered finite
        !! difference stencil of the parallel derivative
        real(kind=GP), allocatable, dimension(:,:) :: prefac_cd_y_p
        !! Prefactor of the k + 1 contribution in the centered finite
        !! difference stencil of the parallel derivative
        real(kind=GP), allocatable, dimension(:,:) :: prefac_cd_y_pp
        !! Prefactor of the k + 2 contribution in the centered finite
        !! difference stencil of the parallel derivative
        real(kind=GP), allocatable, dimension(:,:) :: prefac_hyp_y_mm
        !! Prefactor of the k - 2 contribution in the parallel hyperdiffusion
        real(kind=GP), allocatable, dimension(:,:) :: prefac_hyp_y_m
        !! Prefactor of the k - 1 contribution in the parallel hyperdiffusion
        real(kind=GP), allocatable, dimension(:,:) :: prefac_hyp_y_o
        !! Prefactor of the k contribution in the parallel hyperdiffusion
        real(kind=GP), allocatable, dimension(:,:) :: prefac_hyp_y_p
        !! Prefactor of the k + 1 contribution in the parallel hyperdiffusion
        real(kind=GP), allocatable, dimension(:,:) :: prefac_hyp_y_pp
        !! Prefactor of the k + 2 contribution in the parallel hyperdiffusion
        real(kind=GP), allocatable, dimension(:,:) :: H2
        !! Buffer to store the precalculation of the second order Hamiltonian
    contains
        procedure(apply_interface), deferred :: apply
        procedure, private :: initialize_parent_base
    end type

    interface
        subroutine apply_interface(this, da_f_in, da_phi_in, da_A_par_in, &
                                   da_B_par_in, da_f_out)
            !! Applies the operator to the given input values
            import op_rhs_vlasov_eq_static_base_t, data_array_2d_t, &
                   data_array_5d_t
            class(op_rhs_vlasov_eq_static_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_2d_t), intent(in) :: da_phi_in
            !! Input potential
            class(data_array_2d_t), intent(in) :: da_A_par_in
            !! Input parallel vector potential
            class(data_array_2d_t), intent(in) :: da_B_par_in
            !! Input parallel magnetic fluctuations
            class(data_array_5d_t), intent(inout) :: da_f_out
            !! Output distribution function
        end subroutine

        module subroutine initialize_parent_base(this, mesh)
            !! Initialize the op_rhs_vlasov_eq_static base type
            class(op_rhs_vlasov_eq_static_base_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, abstract, extends(op_rhs_vlasov_eq_static_base_t) :: &
                                                      op_rhs_vlasov_eq_static_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives
        real(kind=GP) :: prefac_cd_vp
        !! Prefactor for the centered finite difference in parallel velspace
        real(kind=GP) :: prefac_hyp_vp
        !! Prefactor for the hyperdiffusion in parallel velspace
        real(kind=GP), allocatable, dimension(:) :: prefac_bps, &
                                                    prefac_orth, &
                                                    prefac_par, &
                                                    prefac_flutter_xyz, &
                                                    prefac_flutter_vp
        !! Species dependent prefactors
    contains
        procedure(initialize_interface), deferred :: initialize
        procedure, private :: initialize_parent
    end type

    interface
        module subroutine initialize_interface(this, mesh, bsg_op)
            !! Initializes the op_rhs_vlasov_eq_static type
            class(op_rhs_vlasov_eq_static_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! BSG operator
        end subroutine

        module subroutine initialize_parent(this, mesh)
            !! Initializes the parent op_rhs_vlasov_eq_static_t type
            class(op_rhs_vlasov_eq_static_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_rhs_vlasov_eq_static_t) :: &
                                                  op_rhs_vlasov_eq_static_cpu_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives on the
        !! CPU
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
        procedure, private :: apply_sg_cpu
        procedure, private :: apply_bsg_cpu
    end type

    interface
        module subroutine initialize_cpu(this, mesh, bsg_op)
            class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                    da_B_par_in, da_f_out)
            class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in
            class(data_array_2d_t), intent(in) :: da_A_par_in
            class(data_array_2d_t), intent(in) :: da_B_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine apply_sg_cpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                    da_B_par_in, da_f_out)
            !! Applies the operator to the given input values for
            !! op_rhs_vlasov_eq_static_cpu_t with structured vp_grid
            class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in
            class(data_array_2d_t), intent(in) :: da_A_par_in
            class(data_array_2d_t), intent(in) :: da_B_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine apply_bsg_cpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                    da_B_par_in, da_f_out)
            !! Applies the operator to the given input values for
            !! op_rhs_vlasov_eq_static_cpu_t with BSG
            class(op_rhs_vlasov_eq_static_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in
            class(data_array_2d_t), intent(in) :: da_A_par_in
            class(data_array_2d_t), intent(in) :: da_B_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, abstract, extends(op_rhs_vlasov_eq_static_base_t) :: &
                                                op_rhs_vlasov_eq_static_vspec_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives, using
        !! spectral method.
        !! TO DO: add B_par
        real(kind=GP), allocatable, dimension(:) :: invsqrt_masses
        !! Inverse sqrt of the masses
        real(kind=GP), pointer, contiguous, dimension(:) :: vp
        !! Vp grid
        real(kind=GP), dimension(:), allocatable :: vpp
        !! Vp + 1 grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vp
        !! Sqrt(vp) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vpp
        !! Sqrt(vp + 1) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vp_o2
        !! Sqrt(vp/2) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vpp_o2
        !! Sqrt((vp + 1)/2) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vp_o2_p
        !! Sqrt(vp/2 + 1)) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vpvpm
        !! Sqrt(vp(vp - 1)) grid
        real(kind=GP), dimension(:), allocatable :: sqrt_vppvppp
        !! Sqrt((vp + 1)*(vp + 2)) grid
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Mu grid
        real(kind=GP), dimension(:), allocatable :: mup
        !! Mu + 1 grid
        real(kind=GP), dimension(:,:), allocatable :: prefac_hyp_vp
        !! Prefactor of the vp hyperdiffusion applied to the
        !! moments with the highest degree
        type(op_set_uniform_cpu_t) :: op_set_uniform
        !! Operator set uniform
    contains
        procedure(initialize_interface_vspec), deferred :: initialize
        procedure, private :: initialize_parent_vspec
    end type

    interface
        module subroutine initialize_interface_vspec(this, mesh)
            !! Initializes the op_rhs_vlasov_eq_static_vspec_t type
            class(op_rhs_vlasov_eq_static_vspec_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

        module subroutine initialize_parent_vspec(this, mesh)
            !! Initializes the parent op_rhs_vlasov_eq_static_vspec_t type
            class(op_rhs_vlasov_eq_static_vspec_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_rhs_vlasov_eq_static_vspec_t) :: &
                                            op_rhs_vlasov_eq_static_vspec_cpu_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives, using
        !! spectral method on CPU
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, mesh)
            class(op_rhs_vlasov_eq_static_vspec_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_vspec_cpu(this, da_f_in, da_phi_in, &
                                          da_A_par_in, da_B_par_in, da_f_out)
            class(op_rhs_vlasov_eq_static_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in
            class(data_array_2d_t), intent(in) :: da_A_par_in
            class(data_array_2d_t), intent(in) :: da_B_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_rhs_vlasov_eq_static_t) :: &
        op_rhs_vlasov_eq_static_gpu_t
        !! Operator to calculate the right-hand side of the gyrokinetic Vlasov
        !! equation excluding terms involving explicit time derivatives on the
        !! GPU
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_rhs_vlasov_eq_static_gpu_t C++ class
        !! instance pointer
    contains
        procedure :: initialize => initialize_gpu
        procedure :: apply => apply_gpu
        final     :: finalize_gpu
    end type

    interface
        module subroutine initialize_gpu(this, mesh, bsg_op)
            class(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_gpu(this, da_f_in, da_phi_in, da_A_par_in, &
                                    da_B_par_in, da_f_out)
            class(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in
            class(data_array_2d_t), intent(in) :: da_A_par_in
            class(data_array_2d_t), intent(in) :: da_B_par_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine finalize_gpu(this)
            type(op_rhs_vlasov_eq_static_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

end module
