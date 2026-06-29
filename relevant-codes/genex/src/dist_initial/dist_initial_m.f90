module dist_initial_m
    !! Module containing different initial distribution functions for the
    !! initialization of the simulation
    !!
    !! Summary of contained distributions:
    !!
    !! dist_initial_t
    !!      1) dist_initial_isotropic_t
    !!          a) dist_initial_maxw(_cpu)_t
    !!          b) dist_initial_maxw_vspec(_cpu)_t
    !!          c) dist_initial_maxw_loc_vspec(_cpu)_t
    !!          d) dist_initial_double_maxw(_cpu)_t
    !!          e) dist_initial_ring(_cpu)_t
    !!          f) dist_initial_slowing_down(_cpu)_t

    !!      2) dist_initial_anisotropic_t
    !!          a) dist_initial_bi_maxw(_cpu)_t

    use genex_fortran_env_m, only: GP
    use profile_container_m, only: profile_container_t

    use params_dist_initial_double_maxw_m, only: &
        params_dist_initial_double_maxw_t
    use params_dist_initial_ring_m, only: &
        params_dist_initial_ring_t
    use params_dist_initial_slowing_down_m, only: &
        params_dist_initial_slowing_down_t
    use params_dist_initial_bi_maxw_m, only: &
        params_dist_initial_bi_maxw_t
    use params_dist_initial_maxw_vspec_m, only: &
        params_dist_initial_maxw_vspec_t

    implicit none
    private

    type, public, abstract :: dist_initial_t
        !! Type defining an initial distribution function of a single species
        !! to initialize the simulation
        type(profile_container_t), private, dimension(:), allocatable :: &
                                                             profile_container
        !! Array of profile containers for all profiles of all species.
        !! We require all profiles because special types of initial
        !! distributions exist that have parameters based on the profiles of all
        !! species.
        integer, private :: spec_ind
        !! Species index of this instance
    contains
        procedure(initialize_interface), deferred, public :: initialize
        !! Initialize initial distribution function type
        procedure(eval_interface), deferred, public :: eval
        !! Evaluate the distribution function
    end type

    interface !dist_initial_t
        subroutine initialize_interface(this, profile_container, &
                                        spec_ind)
            import dist_initial_t, profile_container_t
            !! Initialize the type
            class(dist_initial_t), intent(inout) :: this
            !! Instance of the type
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            !! Profile container
            integer, intent(in) :: spec_ind
            !! Species index
        end subroutine

        real(kind=GP) function eval_interface(this, rho, theta, phi, vp, muB)
            !! Returns the distribution function evaluated in rho-theta-phi
            !! real space and vp-muB velocity space
            import dist_initial_t, GP

            class(dist_initial_t), intent(in) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: rho
            !! Normalized poloidal flux surface label
            real(kind=GP), intent(in) :: theta
            !! Poloidal angle
            real(kind=GP), intent(in) :: phi
            !! Toroidal angle
            real(kind=GP), intent(in) :: vp
            !! Parallel velocity
            real(kind=GP), intent(in) :: muB
            !! Magnetic moment times absolute magnetic field
        end function
    end interface

    type, public, abstract, extends(dist_initial_t) :: dist_initial_isotropic_t
        !! Isotropic temperature distribution, base
        !! This distribution function has one temperature
    contains
        procedure, private :: initialize_isotropic
    end type

    interface !dist_initial_isotropic_t
        module subroutine initialize_isotropic(this, profile_container, &
                                               spec_ind)
            class(dist_initial_isotropic_t), intent(inout) :: this
            !! Instance of the type
            type(profile_container_t), dimension(:), intent(in) :: &
                                                             profile_container
            !! Array of profile containers for all profiles of all species.
            integer, intent(in) :: spec_ind
            !! Species index of this instance
        end subroutine
    end interface

    type, public, abstract, extends(dist_initial_isotropic_t) :: &
                                                             dist_initial_maxw_t
        !! Maxwellian distribution function, base
    end type

    type, public, extends(dist_initial_maxw_t) :: dist_initial_maxw_cpu_t
        !! Maxwellian  distribution function, cpu implementation
    contains
        procedure, public :: initialize => initialize_maxw_cpu
        procedure, public :: eval => eval_maxw_cpu
    end type

    interface !dist_initial_maxw_cpu_t
        module subroutine initialize_maxw_cpu(this, profile_container, &
                                              spec_ind)
            class(dist_initial_maxw_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_maxw_cpu(this, rho, theta, &
                                                    phi, vp, muB)
            class(dist_initial_maxw_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, extends(dist_initial_maxw_t) :: &
                                                   dist_initial_maxw_vspec_cpu_t
        !! Spectral representation of Maxwellian distribution function using
        !! the reference temperature, T_ref, to normalize the basis functions.
        !! As a consequence, the spectral representation includes an explicit
        !! temperature dependence. Cpu implementation.
        type(params_dist_initial_maxw_vspec_t) :: params
        !! Parameter container

    contains
        procedure, public :: initialize => initialize_maxw_vspec_cpu
        procedure, public :: eval => eval_maxw_vspec_cpu
    end type

    interface
        module subroutine initialize_maxw_vspec_cpu(this, &
                                                    profile_container, &
                                                    spec_ind)
            class(dist_initial_maxw_vspec_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_maxw_vspec_cpu(this, rho, &
                                                          theta, phi, &
                                                          vp, muB)
            class(dist_initial_maxw_vspec_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, extends(dist_initial_maxw_t) :: &
                                               dist_initial_maxw_loc_vspec_cpu_t
        !! Spectral representation of Maxwellian distribution function using
        !! the local temperature, T_0\alpha(\psi) from the initial profile,
        !! to normalize the basis functions. As a consequence,
        !! the spectral representation is independent of the
        !! temperature profile. Cpu implementation.
        type(params_dist_initial_maxw_vspec_t) :: params
        !! Parameter container
    contains
        procedure, public :: initialize => initialize_maxw_loc_vspec_cpu
        procedure, public :: eval => eval_maxw_loc_vspec_cpu
    end type

    interface
        module subroutine initialize_maxw_loc_vspec_cpu(this, &
                                                        profile_container, &
                                                        spec_ind)
            class(dist_initial_maxw_loc_vspec_cpu_t), intent(inout) :: &
                                                                        this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                             profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_maxw_loc_vspec_cpu(this, &
                                                              rho, &
                                                              theta, &
                                                              phi, &
                                                              vp, muB)
            class(dist_initial_maxw_loc_vspec_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, abstract, extends(dist_initial_isotropic_t) :: &
                                                      dist_initial_double_maxw_t
        !! Double-Maxwellian distribution function, base
        !! This distribution function describes two added Maxwellians
    end type

    type, public, extends(dist_initial_double_maxw_t) :: &
                                                  dist_initial_double_maxw_cpu_t
        !! Double-Maxwellian distribution function, cpu implementation
        type(params_dist_initial_double_maxw_t) :: params
        !! Parameter container of the dist intial
    contains
        procedure, public :: initialize => initialize_double_maxw_cpu
        procedure, public :: eval => eval_double_maxw_cpu
    end type

    interface !dist_initial_double_maxw_t
        module subroutine initialize_double_maxw_cpu(this, &
                                                     profile_container, &
                                                     spec_ind)
            class(dist_initial_double_maxw_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                              profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_double_maxw_cpu(this, rho, &
                                                           theta, phi, &
                                                           vp, muB)
            class(dist_initial_double_maxw_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, abstract, extends(dist_initial_isotropic_t) :: &
                                                             dist_initial_ring_t
        !! Ring distribution function, base
        !! This distribution function has the name from its ring shape in
        !! vperp1 and vperp2 space. In vp, muB space it is a Gaussian.
    end type

    type, public, extends(dist_initial_ring_t) :: dist_initial_ring_cpu_t
        !! Ring distribution function, cpu implementation
        type(params_dist_initial_ring_t) :: params
        !! Parameter container of the dist intial
    contains
        procedure, public :: initialize => initialize_ring_cpu
        procedure, public :: eval => eval_ring_cpu
    end type

    interface !dist_initial_ring_cpu_t
        module subroutine initialize_ring_cpu(this, profile_container, &
                                              spec_ind)
            class(dist_initial_ring_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_ring_cpu(this, rho, &
                                                    theta, phi, &
                                                    vp, muB)
            class(dist_initial_ring_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, abstract, extends(dist_initial_isotropic_t) :: &
                                                     dist_initial_slowing_down_t
        !! Slowing down distribution function, base
        !! This distribution function describes the slowing down of fast
        !! particles that are "born" at a certain speed.
    end type

    type, public, extends(dist_initial_slowing_down_t) :: &
                                                 dist_initial_slowing_down_cpu_t
        !! Slowing down distribution function, cpu implementation
        type(params_dist_initial_slowing_down_t) :: params
        !! Parameter container of the dist intial
    contains
        procedure, public :: initialize => initialize_slowing_down_cpu
        procedure, public :: eval => eval_slowing_down_cpu
    end type

    interface !dist_initial_slowing_down_cpu_t
        module subroutine initialize_slowing_down_cpu(this, &
                                                      profile_container, &
                                                      spec_ind)
            class(dist_initial_slowing_down_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                              profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_slowing_down_cpu(this, rho, &
                                                            theta, phi, &
                                                            vp, muB)
            class(dist_initial_slowing_down_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

    type, public, abstract, extends(dist_initial_t) :: &
                                                      dist_initial_anisotropic_t
        !! Anisotropic temperature distribution, base
        !! This distribution function has two temperatures,
        !! different in parallel and perperndicular direction
    contains
        procedure, private :: initialize_anisotropic
    end type

    interface !dist_initial_anisotropic_t
        module subroutine initialize_anisotropic(this, profile_container, &
                                                 spec_ind)
            class(dist_initial_anisotropic_t), intent(inout) :: this
            !! Instance of the type
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            !! Array of profile containers for all profiles of all species.
            integer, intent(in) :: spec_ind
            !! Species index of this instance
        end subroutine
    end interface

    type, public, abstract, extends(dist_initial_anisotropic_t) :: &
                                                          dist_initial_bi_maxw_t
        !! Bi-Maxwellian distribution function, base
        !! This distribution function provides the possibility to set up a
        !! Maxwellian with anisotropy in temperature.
        !! Additionally one can use isotropic temperature and the shift
        !! parameter to emulate a drifting Maxwellian.
    end type

    type, public, extends(dist_initial_bi_maxw_t) :: &
                                                     dist_initial_bi_maxw_cpu_t
        !! Bi-Maxwellian distribution function, cpu implementation
        type(params_dist_initial_bi_maxw_t) :: params
        !! Parameter container of the dist intial
    contains
        procedure, public :: initialize => initialize_bi_maxw_cpu
        procedure, public :: eval => eval_bi_maxw_cpu
    end type

    interface !dist_initial_bi_maxw_cpu_t
        module subroutine initialize_bi_maxw_cpu(this, &
                                                 profile_container, &
                                                 spec_ind)
            class(dist_initial_bi_maxw_cpu_t), intent(inout) :: this
            type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
            integer, intent(in) :: spec_ind
        end subroutine
        real(kind=GP) module function eval_bi_maxw_cpu(this, rho, &
                                                             theta, phi, &
                                                             vp, muB)
            class(dist_initial_bi_maxw_cpu_t), intent(in) :: this
            real(kind=GP), intent(in) :: rho, theta, phi, vp, muB
        end function
    end interface

end module
