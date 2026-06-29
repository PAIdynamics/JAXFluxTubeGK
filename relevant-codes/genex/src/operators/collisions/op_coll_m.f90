module op_coll_m
    !! Contains operators to calculate the effect of collisions on the
    !! distribution function.
    !! The Vlasov equation will then become of kinetic type where the eq
    !! df/dt = rhs(f) + coll(f) is solved.
    !! Collision operator applied via this type, take the input distribution
    !! function and the collisional moments (supplied via operators of class
    !! op_mom_coll_t) and add the result to the output distribution function.
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_4d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_COLL, &
                                    GENEX_WRN_COLL

    implicit none

    type, public, abstract, extends(op_base_t) :: op_coll_base_t
        !! Base type for all kinds of collision operators
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        class(data_array_2d_t), private, allocatable :: da_collog
        !! Coulomb logarithm
        logical, private :: is_initialized = .false.
        !! Logical to keep track of the allocation of Coulomb logarithm array
        !! and initialization of the velocity tensor in FPL operator
    contains
        procedure(apply_coll), deferred :: apply
        !! Method to apply the operator to a distribution function
    end type

    interface !op_coll_base_t
        subroutine apply_coll(this, da_f_in, da_moments, da_f_out)
            !! Method to apply the operator to a distribution function
            import op_coll_base_t, data_array_4d_t, data_array_5d_t
            class(op_coll_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_4d_t), intent(inout) :: da_moments
            !! Input collisional moments as (n, V, T) in 3rd dimension
            class(data_array_5d_t), intent(inout) :: da_f_out
            !! Output distribution function (changed by collisions)
        end subroutine
    end interface

    type, public, abstract, extends(op_coll_base_t) :: op_coll_t
        !! Type for all kinds of collision operators using grid approach
        real(kind=GP), dimension(:), private, allocatable :: prefac_bps
        !! Prefactor for B parallel star
    contains
        procedure(initialize_coll), deferred :: initialize
        !! Method to initialize the operator
    end type

    interface !op_coll_t
        module subroutine initialize_coll(this, dcomm_handler, mesh)
            !! Method to initialize the operator
            class(op_coll_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
        end subroutine
    end interface

    type, public, extends(op_coll_t) :: op_coll_bgk_cpu_t
        !! Concrete type that implements collisions with BGK operator on CPU.
        !! BGK: Bhatnagar, Gross, Krook (or commonly just Krook operator)
        !! This operator does relax the distribution function towards a
        !! Maxwellian.
        real(kind=GP), private :: prefac_nu
        !! Collision frequency prefactor
    contains
        procedure :: initialize => initialize_coll_bgk_cpu
        procedure :: apply => apply_coll_bgk_cpu
    end type

    interface !op_coll_bgk_cpu_t
        module subroutine initialize_coll_bgk_cpu(this, dcomm_handler, mesh)
            class(op_coll_bgk_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_bgk_cpu(this, da_f_in, da_moments, &
                                             da_f_out)
            !! This subroutine applies the operator apply_coll_bgk_cpu to the
            !! distribution function f_in. The result will be added to f_out.
            class(op_coll_bgk_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_coll_t) :: op_coll_bgk_gpu_t
        !! Concrete type that implements collisions with BGK operator on GPU.
        !! BGK: Bhatnagar, Gross, Krook (or commonly just Krook operator)
        !! This operator relaxes the distribution function towards a
        !! Maxwellian.
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_coll_bgk_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_coll_bgk_gpu
        procedure :: apply => apply_coll_bgk_gpu
        final     :: finalize_coll_bgk_gpu
    end type

    interface
        module subroutine initialize_coll_bgk_gpu(this, dcomm_handler, mesh)
            class(op_coll_bgk_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_bgk_gpu(this, da_f_in, da_moments, &
                                             da_f_out)
            !! This subroutine applies the operator apply_coll_bgk_gpu to the
            !! distribution function f_in. The result will be added to f_out
            class(op_coll_bgk_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine finalize_coll_bgk_gpu(this)
            !! This destructor also deallocate the operator from GPU memory
            type(op_coll_bgk_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

    type, public, extends(op_coll_t) :: op_coll_lbd_cpu_t
        !! Concrete type that implements the Lenard-Bernstein/Dougherty (LBD)
        !! collision operator in a finite volume discretization.
        !! This is the simplest type of a Fokker Planck operator with a
        !! friction and a diffusion term.
        real(kind=GP), private :: prefac_nu
        !! Collision frequency prefactor
        real(kind=GP), private :: delta_vpar
        !! Grid spacing of vp grid
        real(kind=GP), private :: delta_sqrt_mu
        !! Grid spacing of sqrt mu grid
        real(kind=GP), private :: inv_delta_vpar
        !! Inverse vp grid spacing
        integer, private :: n_vp
        !! Number of vp points
        integer, private :: n_mu
        !! Number of mu points

        class(data_array_4d_t), allocatable, private :: da_boundary_sums
        ! Buffer to store the boundary terms of the corrected moments
        class(data_array_5d_t), allocatable, private :: da_corr_moments
        ! Buffer to store the corrected moments

        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator to set uniform values to arrays
    contains
        procedure :: initialize => initialize_coll_lbd_cpu
        procedure :: apply => apply_coll_lbd_cpu
        procedure, private :: calc_boundary_terms
        procedure, private :: calc_corrected_moments
    end type

    interface !op_coll_lbd_cpu_t
        module subroutine initialize_coll_lbd_cpu(this, dcomm_handler, mesh)
            class(op_coll_lbd_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_lbd_cpu(this, da_f_in, da_moments, &
                                             da_f_out)
            !! This subroutine applies the operator op_coll_lbd_cpu_t to the
            !! distribution function f_in. The result will be added to f_out.
            class(op_coll_lbd_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine calc_boundary_terms(this, da_f_in, da_boundary_sums)
            !! This subroutine calculates the boundary terms needed for the
            !! corrected moments.
            class(op_coll_lbd_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_boundary_sums
        end subroutine

        module subroutine calc_corrected_moments(this, da_moments, &
                                                 da_boundary_sums, &
                                                 da_corr_moments)
            !! This subroutine calculates the corrected moments out of the
            !! default moments of op_coll_m using the boundary terms from
            !! calc_boundary_terms.
            class(op_coll_lbd_cpu_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_moments
            class(data_array_4d_t), intent(in) :: da_boundary_sums
            class(data_array_5d_t), intent(inout) :: da_corr_moments
        end subroutine
    end interface

    type, public, extends(op_coll_t) :: op_coll_lorentz_cpu_t
        !! Concrete type that implements the Lorentz
        !! collision operator in a finite volume discretization.
        !! This operator applies in the limit where m_b >> m_a,
        !! reducing to pure pitch-angle scattering.
        !! In this regime, the collision frequency scales as 1/v^3.
        real(kind=GP), private :: delta_vpar
        !! Grid spacing of vp grid
        real(kind=GP), private :: delta_sqrt_mu
        !! Grid spacing of sqrt mu grid
        real(kind=GP), dimension(:,:), allocatable, private :: inv_v3_stag
        !! Inverse v cubed on staggered grid
        real(kind=GP), dimension(:,:), allocatable, private :: prefac_nu
        !! Collision frequency prefactor for each species pair
        integer, private :: n_vp
        !! Number of vp points
        integer, private :: n_mu
        !! Number of mu points
    contains
        procedure :: initialize => initialize_coll_lorentz_cpu
        procedure :: apply => apply_coll_lorentz_cpu
    end type

    interface !op_coll_lorentz_cpu_t
        module subroutine initialize_coll_lorentz_cpu(this, dcomm_handler, mesh)
            class(op_coll_lorentz_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_lorentz_cpu(this, da_f_in, da_moments, &
                                                 da_f_out)
            !! This subroutine applies op_coll_lorentz_cpu_t to the
            !! distribution function f_in. The result will be added to f_out.
            class(op_coll_lorentz_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_coll_t) :: op_coll_fpl_cpu_t
        !! Concrete type that implements the Fokker-Planck-Landau (FPL)
        !! operator. References:
        !! Yoon (2014) https://doi.org/10.1063/1.4867359
        !! Hager (2016) https://doi.org/10.1016/j.jcp.2016.03.064
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator to set uniform values to arrays

        real(kind=GP), private :: delta_vp
        !! Full spacing of parallel velocity grid
        real(kind=GP), private :: delta_sqrt_mu
        !! Full spacing of sqrt mu grid

        integer, private :: n_sp
        !! Maximum index of species
        integer, private :: n_vp
        !! Maximum index of parallel velocity points
        integer, private :: n_mu
        !! Maximum index of mu points
        integer, private :: n_vp_stag
        !! Maximum index on staggered vp points
        integer, private :: n_mu_stag
        !! Maximum index on staggered mu points

        integer, dimension(3:4), private :: lb_stagger
        !! Lower bounds on staggered grid
        integer, dimension(3:4), private :: ub_stagger
        !! Upper bounds on staggered grid

        real(kind=GP), dimension(:), private, allocatable :: vp_stag
        !! Staggered parallel velocity grid
        real(kind=GP), dimension(:), private, allocatable :: sqrt_mu_stag
        !! Staggered sqrt mu grid

        ! 8 dimensional velocity tensor components. 2 dimensions for real space
        ! and each 2 dimensions for vp, mu and species primed and unprimed.
        real(kind=GP), dimension(:,:,:,:,:,:,:,:), &
                         private, allocatable :: U_par_par
        !! Velocity tensor parallel-parallel component on staggered grid
        real(kind=GP), dimension(:,:,:,:,:,:,:,:), &
                         private, allocatable :: U_par_perpprime
        !! Velocity tensor parallel-perpendicular(primed) component on
        !! staggered grid
        real(kind=GP), dimension(:,:,:,:,:,:,:,:), &
                         private, allocatable :: U_perp_par
        !! Velocity tensor perpendicular-parallel component on staggered grid
        real(kind=GP), dimension(:,:,:,:,:,:,:,:), &
                         private, allocatable :: U_perp_perp
        !! Velocity tensor perpendicular-perpendicular component on
        !! staggered grid
        real(kind=GP), dimension(:,:,:,:,:,:,:,:), &
                         private, allocatable :: U_perp_perpprime
        !! Velocity tensor perpendicular-perpendicular(primed) component on
        !! staggered grid

    contains
        procedure :: initialize => initialize_coll_fpl_cpu
        procedure :: apply => apply_coll_fpl_cpu
        procedure, private :: initialize_staggered_bounds
        procedure, private :: calc_interp_weights
    end type

    interface !op_coll_fpl_cpu_t
        module subroutine initialize_coll_fpl_cpu(this, dcomm_handler, mesh)
            class(op_coll_fpl_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_fpl_cpu(this, da_f_in, da_moments, &
                                             da_f_out)
            !! This subroutine applies the operator op_coll_fpl_cpu_t to the
            !! distribution function f_in. The result will be added to f_out.
            class(op_coll_fpl_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine

        module subroutine initialize_staggered_bounds(this, lb_stripped, &
                                                     ub_stripped)
            !! Initializes the staggered grid boundaries
            class(op_coll_fpl_cpu_t), intent(inout) :: this
            integer, dimension(5), intent(in) :: lb_stripped
            integer, dimension(5), intent(in) :: ub_stripped
        end subroutine

        pure module subroutine calc_interp_weights(this, inv_temp, absB, &
                                                   delta_par, delta_perp, ierr)
            !! Calculates the interpolation weights that preserve the
            !! thermodynamic equilibrium for a single real space point and
            !! species combination for the whole velocity space.
            class(op_coll_fpl_cpu_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: inv_temp
            !! Inverse mixing temperature
            real(kind=GP), intent(in) :: absB
            !! Absolute magnetic field
            real(kind=GP), dimension(1:this%n_vp), intent(out) :: delta_par
            !! Parallel interpolation weights
            real(kind=GP), dimension(1:this%n_mu), intent(out) :: delta_perp
            !! Perpendicular interpolation weights
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

    end interface

    interface
        module subroutine calc_collog(da_moments, da_collog)
            !! Method to calculate the Coulomb logarithm
            class(data_array_4d_t), intent(in) :: da_moments
            !! Input collisional moments as (n, V, T) in 3rd dimension
            class(data_array_2d_t), intent(inout) :: da_collog
            !! Output array containing values of the Coulomb logarithm
        end subroutine
    end interface

    type, public, abstract, extends(op_coll_base_t) :: op_coll_vspec_t
        !! Type for all kinds of collision operators using spectral approach
        real(kind=GP), contiguous, pointer, dimension(:) :: vp
        !! Pointer to vp spectral grid
        real(kind=GP), contiguous, pointer, dimension(:) :: mu
        !! Pointer to mu spectral grid
        integer, private :: n_sp
        !! Maximum index of species
        real(kind=GP), private :: prefac_nu
        !! Collision frequency prefactor
        real(kind=GP), private, allocatable, dimension(:) :: temp_scalings
        !! Temperature scaling
    contains
        procedure(initialize_coll_vspec), deferred :: initialize
        !! Method to initialize the spectral operator
    end type

    interface
        module subroutine initialize_coll_vspec(this, dcomm_handler, mesh)
            !! Method to initialize the operator
            class(op_coll_vspec_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
        end subroutine
    end interface

    type, public, extends(op_coll_vspec_t) :: op_coll_lbd_vspec_cpu_t
        !! Concrete type that implements the Lenard-Bernstein/Dougherty (LBD)
        !! collision operator using spectral approach, CPU implementation.
        real(kind=GP), dimension(:), allocatable :: sqrt_2vp
        !! Numerical prefactors sqrt(2 * vp)
        real(kind=GP), dimension(:), allocatable :: sqrt_vpvpm
        !! Numerical prefactors sqrt(vp * (vp - 1))
    contains
        procedure :: initialize => initialize_coll_lbd_vspec_cpu
        procedure :: apply => apply_coll_lbd_vspec_cpu
    end type

    interface ! op_coll_lbd_vspec_cpu_t
        module subroutine initialize_coll_lbd_vspec_cpu(this, dcomm_handler, &
                                                        mesh)
            class(op_coll_lbd_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coll_lbd_vspec_cpu(this, da_f_in, da_moments, &
                                                   da_f_out)
            class(op_coll_lbd_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

end module
