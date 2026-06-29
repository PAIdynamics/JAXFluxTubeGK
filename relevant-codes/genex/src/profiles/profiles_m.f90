module profiles_m
    !! Module containing the implementation of different profiles to initialize
    !! the simulation and to set Dirichlet boundary conditions
    !!
    !! Summary of contained profiles:
    !!
    !! profile_t
    !!      1) profile_uniform(_cpu)_t
    !!      2) profile_cbc(_cpu)_t
    !!      3) profile_sine(_cpu)_t
    !!      4) profile_lapd(_cpu)_t
    !!      4) profile_plateau_sine(_cpu)_t

    use genex_fortran_env_m, only : GP
    use mesh_5d_m, only: mesh_5d_t
    use params_profile_uniform_m, only: params_profile_uniform_t
    use params_profile_cbc_m, only: params_profile_cbc_t
    use params_profile_sine_m, only: params_profile_sine_t
    use params_profile_lapd_m, only: params_profile_lapd_t
    use params_profile_plateau_m, only: params_profile_plateau_t
    implicit none
    private

    type, public, abstract :: profile_t
        !! Type defining a profile to initialize the simulation
    contains
        procedure(eval_interface), deferred, public :: eval
    end type

    abstract interface !profile_t
        real(kind = GP) function eval_interface(this, rho, theta, phi)
            !! Returns the value of the profile at the given position
            import profile_t, GP
            class(profile_t), intent(in) :: this
            !! Instance of the type
            real(kind = GP), intent(in) :: rho
            !! Normalized poloidal flux surface label
            real(kind = GP), intent(in) :: theta
            !! Poloidal angle
            real(kind = GP), intent(in) :: phi
            !! Toroidal angle
        end function
    end interface

    type, public, abstract, extends(profile_t) :: profile_uniform_t
        !! Uniform profile, base
    end type

    type, public, extends(profile_uniform_t) :: profile_uniform_cpu_t
        !! Uniform profile, cpu implementation
        type(params_profile_uniform_t) :: params
        !! Parameter container of the profile
    contains
        procedure, public :: initialize => initialize_uniform_cpu
        procedure, public :: eval => eval_uniform_cpu
    end type

    interface !profile_uniform_cpu_t
        module subroutine initialize_uniform_cpu(this, params)
            class(profile_uniform_cpu_t), intent(inout) :: this
            class(params_profile_uniform_t), intent(inout) :: params
        end subroutine
        real(kind=GP) module function eval_uniform_cpu(this, rho, theta, phi)
            class(profile_uniform_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi
        end function
    end interface

    type, public, abstract, extends(profile_t) :: profile_cbc_t
        !! Cyclotron base case profile described in
        !! Lapillone et al, Physics of Plasmas 17, 112321 (2010) and
        !! Falchetto et al, Plasma Phys. Control. Fusion 50 124015 (2008)
    end type

    type, public, extends(profile_cbc_t) :: profile_cbc_cpu_t
        !! Cyclotron base case profile, cpu implementation
        type(params_profile_cbc_t) :: params
        !! Parameter container of the profile
    contains
        procedure, public :: initialize => initialize_cbc_cpu
        procedure, public :: eval => eval_cbc_cpu
    end type

    interface !profile_cbc_cpu_t
        module subroutine initialize_cbc_cpu(this, params)
            class(profile_cbc_cpu_t), intent(inout) :: this
            class(params_profile_cbc_t), intent(inout) :: params
        end subroutine

        real(kind=GP) module function eval_cbc_cpu(this, rho, theta, phi)
            class(profile_cbc_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi
        end function
    end interface

    type, public, abstract, extends(profile_t) :: profile_sine_t
        !! Piecewise profile based on a sine function of the form
        !!
        !! const                        if rho < rho_min
        !! const                        if rho > rho_max
        !! A * sin(a * rho + b) + B     else
        !!
        !! The profile function is differentiable. This profile is well suited
        !! to specify fixed core and scrape-off layer temperatures and
        !! densities and continuously interpolate in between.
    end type

    type, public, extends(profile_sine_t) :: profile_sine_cpu_t
        !! Profile based on a sine function, cpu implementation
        type(params_profile_sine_t) :: params
        !! Parameter container of the profile
    contains
        procedure, public :: initialize => initialize_sine_cpu
        procedure, public :: eval => eval_sine_cpu
    end type

    interface !profile_sine_cpu_t
        module subroutine initialize_sine_cpu(this, params)
            class(profile_sine_cpu_t), intent(inout) :: this
            class(params_profile_sine_t), intent(inout) :: params
        end subroutine

        real(kind=GP) module function eval_sine_cpu(this, rho, theta, phi)
            class(profile_sine_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi
        end function
    end interface

    type, public, abstract, extends(profile_t) :: profile_lapd_t
        !! Profile for LAPD simulations described in
        !! Pan et al, Physics of Plasmas 25, 062303 (2018)
    end type

    type, public, extends(profile_lapd_t) :: profile_lapd_cpu_t
        !! LAPD profile, cpu implementation
        class(mesh_5d_t), pointer :: mesh
        !! Pointer to the mesh
        type(params_profile_lapd_t) :: params
        !! Parameter container of the profile
        logical :: is_density
        !! Indicates if profile is a density or a temperature
        logical :: is_electrons
        !! Indicates if profile is for electrons
    contains
        procedure, public :: initialize => initialize_lapd_cpu
        procedure, public :: eval => eval_lapd_cpu
    end type

    interface !profile_lapd_cpu_t
        module subroutine initialize_lapd_cpu(this, mesh, params, is_density, &
                                              is_electrons)
            class(profile_lapd_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(params_profile_lapd_t), intent(inout) :: params
            logical, intent(in):: is_density
            logical, intent(in):: is_electrons
        end subroutine

        real(kind=GP) module function eval_lapd_cpu(this, rho, theta, phi)
            class(profile_lapd_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi
        end function
    end interface

    type, public, abstract, extends(profile_t) :: profile_plateau_sine_t
        !! Piecewise profile based on two sine step function of the form
        !!
        !! const                        if rho < rho_inner_foot
        !! const                        if rho_inner_top <= rho < rho_outer_top
        !! const                        if rho >= rho_outer_foot
        !! A * sin(a * rho + b) + B     if rho_inner_foot <= rho < rho_inner_top
        !!                              or rho_outer_top <= rho < rho_outer_foot
        !!
        !! The profile function is differentiable. This profile is well suited
        !! to initialize neutrals.
    end type

    type, public, extends(profile_plateau_sine_t) :: profile_plateau_sine_cpu_t
        !! Profile based on a plateau with two sine steps, cpu implementation
        type(params_profile_plateau_t) :: params
        !! Parameter container of the profile
    contains
        procedure, public :: initialize => initialize_plateau_sine_cpu
        procedure, public :: eval => eval_plateau_sine_cpu
    end type

    interface !profile_plateau_sine_cpu_t
        module subroutine initialize_plateau_sine_cpu(this, params)
            class(profile_plateau_sine_cpu_t), intent(inout) :: this
            class(params_profile_plateau_t), intent(inout) :: params
        end subroutine

        real(kind=GP) module function eval_plateau_sine_cpu(this, &
                                                            rho, theta, phi)
            class(profile_plateau_sine_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi
        end function
    end interface
end module
