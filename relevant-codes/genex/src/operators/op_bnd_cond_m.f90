module op_bnd_cond_m
    !! Module containing operators to set boundary condition to the
    !! distribution function, the electrostatic potential,
    !! the parallel component of the electromagnetic vector potential
    !! and the parallel electric field.
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_bnd_cond_base_t
        !! Operator to apply boundary conditions to the distribution function
        !! and the electromagnetic fields
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        integer, private :: max_num_compute_buf_core
        !! Maximum number of compute points in the inner boundary buffer region.
        !! Only defined in the case of Neumann boundary condition.
        integer, private :: max_num_ghost_core
        !! Maximum number of ghost points in the core region.
        !! Only defined in the case of Neumann boundary condition.
        real(kind=GP), allocatable, dimension(:) :: number_bnd_points
        !! Number of boundary points in the inner buffer region per poloidal
        !! plane.Only allocated in the case of Neumann boundary condition.
        integer, allocatable, dimension(:,:), private :: ind_compute_buf_core
        !! RZ indices of compute points in the inner boundary buffer region per
        !! poloidal plane. Only allocated in the case of Neumann boundary
        !! condition.
        integer, allocatable, dimension(:,:), private :: ind_ghost_core
        !! RZ indices of ghost points in the core region per poloidal plane.
        !! Only allocated in the case of Neumann boundary condition.
    contains
        procedure(apply_interface), deferred :: apply
        procedure(initialize_interface), deferred :: initialize
        procedure, private :: initialize_neum_parent
    end type

    abstract interface
        subroutine apply_interface(this, da_f_inout, da_co_qn_eq, &
                                   da_b_qn_eq, da_b_amps_law, da_b_ohms_law, t)
            !! Applies the operator to the given input values
            import op_bnd_cond_base_t, data_array_2d_t, data_array_5d_t, GP
            class(op_bnd_cond_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            !! Polarization
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            !! Electrostatic potential
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            !! Parallel vector potential
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            !! Parallel electric field
            real(kind=GP), intent(in) :: t
            !! Current time
        end subroutine

        subroutine initialize_interface(this, dcomm_handler, mesh)
            !! Initializes the operator
            import op_bnd_cond_base_t, dcomm_handler_t, mesh_5d_t, GP
            class(op_bnd_cond_base_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    interface
        module subroutine initialize_neum_parent(this, dcomm_handler, mesh)
            !! Initializes the parent op_bnd_cond_neum_t type
            class(op_bnd_cond_base_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine
    end interface

    type, public, abstract, extends(op_bnd_cond_base_t) :: op_bnd_cond_dir_t
        !! Operator to apply Dirichlet boundary conditions to the distribution
        !! function and the electromagnetic fields. The Dirichtlet boundary
        !! condition of the distribution function is set according to a given
        !! profile. The Dirichtlet boundary condition for the electric
        !! potential is set according to the sheath boundary conditions.
        !! This operator can be used in both the grid and spectral approach
        !! since the same Dirichelt boundary conditions are assumed.
        !!
        !! NOTE: Currently the Dirichlet boundary condition for the
        !!       distribution is not set explicitly in the operator. The value
        !!       of the distribution function on the boundary is constant in
        !!       in time and set upon initialization with the initial
        !!       condition operator.
    end type

    type, public, extends(op_bnd_cond_dir_t) :: op_bnd_cond_dir_cpu_t
        !! Concrete type that implements Dirichlet boundary condition operator
        !! on CPU
    contains
        procedure, public :: initialize => initialize_dir_cpu
        procedure, public :: apply => apply_dir_cpu
    end type

    interface
        module subroutine initialize_dir_cpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_dir_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_dir_cpu(this, da_f_inout, da_co_qn_eq, &
                                        da_b_qn_eq, da_b_amps_law, &
                                        da_b_ohms_law, t)
            class(op_bnd_cond_dir_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_bnd_cond_dir_t) :: op_bnd_cond_dir_gpu_t
        !! Concrete type that implements Dirichlet boundary condition operator
        !! on GPU
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_bnd_cond_dir_gpu_t C++ class instance pointer
    contains
        procedure, public :: initialize => initialize_dir_gpu
        procedure, public :: apply => apply_dir_gpu
        final             :: finalize_dir_gpu
    end type

    interface
        module subroutine initialize_dir_gpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_dir_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_dir_gpu(this, da_f_inout, da_co_qn_eq, &
                                        da_b_qn_eq, da_b_amps_law, &
                                        da_b_ohms_law, t)
            class(op_bnd_cond_dir_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine

        module subroutine finalize_dir_gpu(this)
            !! This destructor also deallocate the operator from GPU memory
            type(op_bnd_cond_dir_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

    type, public, abstract, extends(op_bnd_cond_base_t) :: op_bnd_cond_neum_t
        !! This operator applies Neumann boundary conditions at the
        !! inner boundary and Dirichlet boundary conditions at the
        !! outer boundary to the distribution function. The Neumann boundary
        !! conditions are enforced by setting the ghost points to the average
        !! over the inner buffer regions, while Dirichlet boundary conditions
        !! are imposed at the outer boundary. Electromagnetic fields remain
        !! unchanged by this operator.
        !!
        !! Note: Currently, the Dirichlet boundary condition for the
        !!       distribution function at the outer boundary is not
        !!       explicitly set within the operator and is fixed by the
        !!       initial conditions.
    end type

    type, public, extends(op_bnd_cond_neum_t) :: op_bnd_cond_neum_cpu_t
        !! This operator applies Neumann boundary conditions at the
        !! inner boundary and Dirichlet boundary conditions at the
        !! outer boundary to the distribution function on the CPU.
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator to initialize buffers
        real(kind=GP), pointer, contiguous, dimension(:) :: vp
        !! Pointer to vp grid
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Pointer to mu grid
    contains
        procedure, public :: initialize => initialize_neum_cpu
        procedure, public :: apply => apply_neum_cpu
    end type

    interface
        module subroutine initialize_neum_cpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_neum_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_neum_cpu(this, da_f_inout, da_co_qn_eq, &
                                         da_b_qn_eq, da_b_amps_law, &
                                         da_b_ohms_law, t)
            class(op_bnd_cond_neum_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_bnd_cond_neum_t) :: op_bnd_cond_neum_gpu_t
        !! This operator applies Neumann boundary conditions at the
        !! inner boundary and Dirichlet boundary conditions at the
        !! outer boundary to the distribution function on the GPU.
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_bnd_cond_dir_gpu_t C++ class instance pointer
    contains
        procedure, public :: initialize => initialize_neum_gpu
        procedure, public :: apply => apply_neum_gpu
        final             :: finalize_neum_gpu
    end type

    interface
        module subroutine initialize_neum_gpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_neum_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_neum_gpu(this, da_f_inout, da_co_qn_eq, &
                                         da_b_qn_eq, da_b_amps_law, &
                                         da_b_ohms_law, t)
            class(op_bnd_cond_neum_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine

        module subroutine finalize_neum_gpu(this)
            type(op_bnd_cond_neum_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

    type, public, abstract, extends(op_bnd_cond_base_t) :: &
                                                        op_bnd_cond_neum_vspec_t
        !! This operator applies Neumann boundary conditions at the
        !! inner boundary and Dirichlet boundary conditions at the
        !! outer boundary to the spectral distribution function.
        !! The Neumann boundary conditions are enforced
        !! by averaging over the inner buffer and ghost regions, while
        !! Dirichlet boundary conditions are imposed
        !! at the outer boundary. Electromagnetic fields remain
        !! unchanged by this operator. If specified, the density and/or
        !! even spectral moments can be frozen at the inner boundary.
        !!
        !! Note: Currently, the Dirichlet boundary condition for the
        !!       distribution function at the outer boundary is not
        !!       explicitly set within the operator and is fixed by the
        !!       initial conditions.
    end type

    type, public, extends(op_bnd_cond_neum_vspec_t) :: &
                                                    op_bnd_cond_neum_vspec_cpu_t
        !! This operator applies Neumann boundary conditions at the
        !! inner boundary and Dirichlet boundary conditions at the
        !! outer boundary to the spectral distribution function on the CPU.
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator to initialize buffers
        real(kind=GP), pointer, contiguous, dimension(:) :: vp
        !! Pointer to vp grid
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Pointer to mu grid
        logical :: freeze_dens
        !! Flag to freeze inner density boundary conditions.
        logical :: freeze_even_mom
        !! Flag to freeze inner even moment boundary conditions.
    contains
        procedure, public :: initialize => initialize_neum_vspec_cpu
        procedure, public :: apply => apply_neum_vspec_cpu
    end type

    interface
        module subroutine initialize_neum_vspec_cpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_neum_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_neum_vspec_cpu(this, da_f_inout, da_co_qn_eq, &
                                               da_b_qn_eq, da_b_amps_law, &
                                               da_b_ohms_law, t)
            class(op_bnd_cond_neum_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine
    end interface

    type, public, abstract, extends(op_bnd_cond_base_t) :: op_bnd_cond_mms_t
        !! Operator to apply boundary conditions for the MMS verficiation of
        !! the code for the grid approach.
        real(kind=GP), private, contiguous, pointer, dimension(:) :: phi, vp, mu
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: R, Z
        real(kind=GP), private, allocatable, dimension(:,:) :: vp_total
        !! Array for the total amount of vp points.
        !! This includes values on the grid and on the boundaries.
        !! Generalized for BSG
    end type

    type, public, extends(op_bnd_cond_mms_t) :: op_bnd_cond_mms_cpu_t
        real(kind=GP), private :: prefac_co_qn_ions
        ! Prefactor for the polarization contribution from ions
        real(kind=GP), private :: prefac_co_qn_electrons
        ! Prefactor for the polarization contribution from electrons
    contains
        procedure, public :: initialize => initialize_mms_cpu
        procedure, public :: apply => apply_mms_cpu
    end type

    interface
        module subroutine initialize_mms_cpu(this, dcomm_handler, mesh)
            class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_mms_cpu(this, da_f_inout, da_co_qn_eq, &
                                        da_b_qn_eq, da_b_amps_law, &
                                        da_b_ohms_law, t)
            class(op_bnd_cond_mms_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            real(kind=GP), intent(in) :: t
        end subroutine
    end interface

end module op_bnd_cond_m
