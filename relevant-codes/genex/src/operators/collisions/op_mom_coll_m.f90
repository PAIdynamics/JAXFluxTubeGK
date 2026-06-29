module op_mom_coll_m
    !! Contains operators that calculate the velocity space moments for the
    !! use in the implementation of collision operators
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

    private

    type, public, abstract, extends(op_base_t) :: op_mom_coll_base_t
        !! Operator to calculate the velocity space moments
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        integer, private :: n_sp
        !! Number of species
    contains
       procedure(initialize_interface), deferred, public :: initialize
       !! Method to initialize the operator
       procedure(apply_interface), deferred, public :: apply
       !! Method to apply the operator
    end type

    abstract interface !op_mom_coll_t
        subroutine initialize_interface(this, dcomm_handler, mesh)
            !! Initializes the op_mom_coll_base_t type
            import op_mom_coll_base_t, dcomm_handler_t, mesh_5d_t
            class(op_mom_coll_base_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
        subroutine apply_interface(this, da_f_in, da_moments)
            !! Calculates the 0th, 1st and 2nd moment for each real-space point
            !! and each species.
            import op_mom_coll_base_t, data_array_4d_t, data_array_5d_t
            class(op_mom_coll_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_4d_t), intent(inout) :: da_moments
            !! Output moments
            !! The 0th, 1st and 2nd moment are stored in the 3rd index
        end subroutine
    end interface

    type, public, abstract, extends(op_mom_coll_base_t) :: op_mom_coll_t
        !! Operator to calculate the velocity space moments using grid approach
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Prefactor for B parallel star
    end type

    type, public, extends(op_mom_coll_t) :: op_mom_coll_cpu_t
        !! Operator to calculate the velocity space moments using
        !! grid approach, CPU implementation
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator for initializing buffers
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface !op_mom_coll_cpu_t
        module subroutine initialize_cpu(this, dcomm_handler, mesh)
            class(op_mom_coll_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_cpu(this, da_f_in, da_moments)
            class(op_mom_coll_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_mom_coll_t) :: op_mom_coll_gpu_t
        !! Operator to calculate the velocity space moments using
        !! grid approach, GPU implementation
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_mom_coll_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_gpu
        procedure :: apply => apply_gpu
        final     :: finalize_gpu
    end type

    interface !op_mom_coll_gpu_t
        module subroutine initialize_gpu(this, dcomm_handler, mesh)
            class(op_mom_coll_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_gpu(this, da_f_in, da_moments)
            class(op_mom_coll_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
        end subroutine

        module subroutine finalize_gpu(this)
            !! This destructor also deallocate the operator from GPU memory
            type(op_mom_coll_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

    type, public, abstract, extends(op_mom_coll_base_t) ::  op_mom_coll_vspec_t
        !! Operator to calculate the velocity space moments using spectral
        !! approach
        !!
        !! NOTE: This type does not depend on the reference profiles
        !!       since we need physical quantities normalized by the
        !!       reference temperature and density in the collision operator.
        real(kind=GP), pointer, contiguous, dimension(:, :, :) :: &
                                                               mom_weights_vspec
        !! Pointer to moment spectral weights
    end type

    type, public, extends(op_mom_coll_vspec_t) :: op_mom_coll_vspec_cpu_t
        !! Operator to calculate the velocity space moments using spectral
        !! approach, CPU implementation
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
            class(op_mom_coll_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine
        module subroutine apply_vspec_cpu(this, da_f_in, da_moments)
            class(op_mom_coll_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
        end subroutine
    end interface

end module op_mom_coll_m
