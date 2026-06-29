module equilibrium_reference_m
    !! Module containing analytical reference values for the magnetic
    !! equilibrium and various derivatives of it for unit testing

    use genex_fortran_env_m, only : GP
    use math_m, only: PI
    implicit none

    type, public, abstract :: equilibrium_reference_t
    contains
        procedure(generic_getter), public, deferred :: absB
        procedure(generic_getter), public, deferred :: normb_x
        procedure(generic_getter), public, deferred :: normb_y
        procedure(generic_getter), public, deferred :: normb_z
        procedure(generic_getter), public, deferred :: curl_normb_y
        procedure(generic_getter), public, deferred :: dabsBdx
        procedure(generic_getter), public, deferred :: dabsBdy
        procedure(generic_getter), public, deferred :: dabsBdz
        procedure(generic_getter), public, deferred :: dgyzdx
        procedure(generic_getter), public, deferred :: dgyxdz
        procedure(generic_getter), public, deferred :: dgyxdx
        procedure(generic_getter), public, deferred :: dgyzdz
        procedure(generic_getter), public, deferred :: dgyxdy
        procedure(generic_getter), public, deferred :: dgyzdy
        procedure(generic_getter), public, deferred :: rho
        procedure(generic_getter), public, deferred :: theta
        procedure(generic_getter), public, deferred :: psi
        procedure(generic_getter), public, deferred :: jacobian
    end type

    abstract interface
        real(kind=GP) function generic_getter(this, x, z, phi)
            import equilibrium_reference_t, GP
            class(equilibrium_reference_t), intent(in) :: this
            real(kind=GP), intent(in) :: x
            real(kind=GP), intent(in) :: z
            real(kind=GP), intent(in) :: phi
        end function
    end interface

    type, public, extends(equilibrium_reference_t) :: slab_reference_t
    contains
        procedure, public :: absB => absB_slab
        procedure, public :: normb_x => normb_x_slab
        procedure, public :: normb_y => normb_y_slab
        procedure, public :: normb_z => normb_z_slab
        procedure, public :: curl_normb_y => curl_normb_y_slab
        procedure, public :: dabsBdx => dabsBdx_slab
        procedure, public :: dabsBdy => dabsBdy_slab
        procedure, public :: dabsBdz => dabsBdz_slab
        procedure, public :: dgyzdx => dgyzdx_slab
        procedure, public :: dgyxdz => dgyxdz_slab
        procedure, public :: dgyxdx => dgyxdx_slab
        procedure, public :: dgyzdz => dgyzdz_slab
        procedure, public :: dgyxdy => dgyxdy_slab
        procedure, public :: dgyzdy => dgyzdy_slab
        procedure, public :: rho => rho_slab
        procedure, public :: theta => theta_slab
        procedure, public :: psi => psi_slab
        procedure, public :: jacobian => jacobian_slab
    end type

    type, public, extends(equilibrium_reference_t) :: circular_reference_t
        real(kind=GP), private :: q_ref, shear, rho_ref
    contains
        procedure, private :: q => q_circular
        procedure, private :: D => D_circular
        procedure, private :: dqdx => dqdx_circular
        procedure, private :: dqdz => dqdz_circular
        procedure, public :: initialize => initialize_circular
        procedure, public :: absB => absB_circular
        procedure, public :: normb_x => normb_x_circular
        procedure, public :: normb_y => normb_y_circular
        procedure, public :: normb_z => normb_z_circular
        procedure, public :: curl_normb_y => curl_normb_y_circular
        procedure, public :: dabsBdx => dabsBdx_circular
        procedure, public :: dabsBdy => dabsBdy_circular
        procedure, public :: dabsBdz => dabsBdz_circular
        procedure, public :: dgyzdx => dgyzdx_circular
        procedure, public :: dgyxdz => dgyxdz_circular
        procedure, public :: dgyxdx => dgyxdx_circular
        procedure, public :: dgyzdz => dgyzdz_circular
        procedure, public :: dgyxdy => dgyxdy_circular
        procedure, public :: dgyzdy => dgyzdy_circular
        procedure, public :: rho => rho_circular
        procedure, public :: theta => theta_circular
        procedure, public :: psi => psi_circular
        procedure, public :: jacobian => jacobian_circular
    end type

    type, public, extends(equilibrium_reference_t) :: salpha_reference_t
        real(kind=GP), private :: q_ref, shear, minor_r, L_ref, B_ref
    contains
        procedure, private :: q => q_salpha
        procedure, private :: D => D_salpha
        procedure, private :: dqdx => dqdx_salpha
        procedure, private :: dqdz => dqdz_salpha
        procedure, public :: initialize => initialize_salpha
        procedure, public :: absB => absB_salpha
        procedure, public :: normb_x => normb_x_salpha
        procedure, public :: normb_y => normb_y_salpha
        procedure, public :: normb_z => normb_z_salpha
        procedure, public :: curl_normb_y => curl_normb_y_salpha
        procedure, public :: dabsBdx => dabsBdx_salpha
        procedure, public :: dabsBdy => dabsBdy_salpha
        procedure, public :: dabsBdz => dabsBdz_salpha
        procedure, public :: dgyzdx => dgyzdx_salpha
        procedure, public :: dgyxdz => dgyxdz_salpha
        procedure, public :: dgyxdx => dgyxdx_salpha
        procedure, public :: dgyzdz => dgyzdz_salpha
        procedure, public :: dgyxdy => dgyxdy_salpha
        procedure, public :: dgyzdy => dgyzdy_salpha
        procedure, public :: rho => rho_salpha
        procedure, public :: theta => theta_salpha
        procedure, public :: psi => psi_salpha
        procedure, public :: jacobian => jacobian_salpha
    end type

    type, public, extends(equilibrium_reference_t) :: dommaschk_reference_t
    contains
        procedure, public :: absB => absB_domm
        procedure, public :: normb_x => normb_x_domm
        procedure, public :: normb_y => normb_y_domm
        procedure, public :: normb_z => normb_z_domm
        procedure, public :: curl_normb_y => curl_normb_y_domm
        procedure, public :: dabsBdx => dabsBdx_domm
        procedure, public :: dabsBdy => dabsBdy_domm
        procedure, public :: dabsBdz => dabsBdz_domm
        procedure, public :: dgyzdx => dgyzdx_domm
        procedure, public :: dgyxdz => dgyxdz_domm
        procedure, public :: dgyxdx => dgyxdx_domm
        procedure, public :: dgyzdz => dgyzdz_domm
        procedure, public :: dgyxdy => dgyxdy_domm
        procedure, public :: dgyzdy => dgyzdy_domm
        procedure, public :: rho => rho_domm
        procedure, public :: theta => theta_domm
        procedure, public :: psi => psi_domm
        procedure, public :: jacobian => jacobian_domm
    end type

contains

    ! slab implementations

    real(kind=GP) function absB_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        absB_slab = 1.0_GP
    end function

    real(kind=GP) function normb_x_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        normb_x_slab = 0.0_GP
    end function

    real(kind=GP) function normb_y_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        normb_y_slab = 1.0_GP
    end function

    real(kind=GP) function normb_z_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        normb_z_slab = 0.0_GP
    end function

    real(kind=GP) function curl_normb_y_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        curl_normb_y_slab = 0.0_GP
    end function

    real(kind=GP) function dabsBdx_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dabsBdx_slab = 0.0_GP
    end function

    real(kind=GP) function dabsBdy_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dabsBdy_slab = 0.0_GP
    end function

    real(kind=GP) function dabsBdz_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dabsBdz_slab = 0.0_GP
    end function

    real(kind=GP) function dgyzdx_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyzdx_slab = 0.0_GP
    end function

    real(kind=GP) function dgyxdz_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyxdz_slab = 0.0_GP
    end function

    real(kind=GP) function dgyzdz_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyzdz_slab = 0.0_GP
    end function

    real(kind=GP) function dgyxdx_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyxdx_slab = 0.0_GP
    end function

    real(kind=GP) function dgyxdy_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyxdy_slab = 0.0_GP
    end function

    real(kind=GP) function dgyzdy_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        dgyzdy_slab = 0.0_GP
    end function

    real(kind=GP) function rho_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        rho_slab = x
    end function

    real(kind=GP) function theta_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        theta_slab = z
    end function

    real(kind=GP) function jacobian_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        jacobian_slab = 1.0_GP
    end function

    real(kind=GP) function psi_slab(this, x, z, phi)
        class(slab_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        psi_slab = 0.0_GP
    end function

    ! circular implementations

    subroutine initialize_circular(this, q_ref, shear, rho_ref)
        class(circular_reference_t), intent(inout) :: this
        real(kind=GP), intent(in) :: q_ref, shear, rho_ref

        this%q_ref = q_ref
        this%shear = shear
        this%rho_ref = rho_ref
    end subroutine

    real(kind=GP) function rho_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        rho_circular = sqrt(x**2 + z**2)
    end function

    real(kind=GP) function q_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        q_circular = this%q_ref / (1.0_GP - this%shear &
                                * (this%rho(x, z, phi) - this%rho_ref))
    end function

    real(kind=GP) function D_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        D_circular = sqrt(x**2 + z**2 + (this%q(x, z, phi)**2))
    end function

    real(kind=GP) function dqdx_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dqdx_circular =  this%shear * x / this%rho(x, z, phi) * this%q_ref &
            / (1.0_GP - this%shear * (this%rho(x, z, phi) - this%rho_ref))** 2
    end function

    real(kind=GP) function dqdz_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dqdz_circular =  this%shear * z / this%rho(x, z, phi) * this%q_ref &
            / (1.0_GP - this%shear * (this%rho(x, z, phi) - this%rho_ref)) ** 2
    end function

    real(kind=GP) function absB_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        absB_circular = sqrt(this%rho(x, z, phi)**2 &
                             / this%q(x, z, phi)**2 + 1.0_GP)
    end function

    real(kind=GP) function normb_x_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_x_circular = -z / this%D(x, z, phi)
    end function

    real(kind=GP) function normb_y_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_y_circular = this%q(x, z, phi) / this%D(x, z, phi)
    end function

    real(kind=GP) function normb_z_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_z_circular = x / this%D(x, z, phi)
    end function

    real(kind=GP) function dgyzdx_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx

        D = this%D(x, z, phi)
        dqdx = this%dqdx(x, z, phi)
        dgyzdx_circular = 1.0_GP / D - x / D**3 * (x + this%q(x, z, phi) * dqdx)
    end function

    real(kind=GP) function dgyxdz_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz

        dqdz = this%dqdz(x, z, phi)
        D = this%D(x, z, phi)
        dgyxdz_circular = -1.0_GP / D &
                        + z / D**3 * (z + this%q(x, z, phi) * dqdz)
    end function

    real(kind=GP) function dgyxdx_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx

        dqdx = this%dqdx(x, z, phi)
        D = this%D(x, z, phi)
        dgyxdx_circular = z * (this%q(x, z, phi) * dqdx + x) / D**3
    end function

    real(kind=GP) function dgyzdz_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz

        dqdz = this%dqdz(x, z, phi)
        D = this%D(x, z, phi)
        dgyzdz_circular =  -x * (this%q(x, z, phi) * dqdz + z) / D**3
    end function

    real(kind=GP) function dgyxdy_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dgyxdy_circular = this%normb_x(x, z, phi) * this%dgyxdx(x, z, phi) &
                        + this%normb_z(x, z, phi) * this%dgyxdz(x, z, phi)
    end function

    real(kind=GP) function dgyzdy_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dgyzdy_circular = this%normb_x(x, z, phi) * this%dgyzdx(x, z, phi) &
                        + this%normb_z(x, z, phi) * this%dgyzdz(x, z, phi)
    end function

    real(kind=GP) function curl_normb_y_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        curl_normb_y_circular = 1.0_GP / abs(this%normb_y(x, z, phi)) * ( &
            + this%normb_x(x, z, phi) * this%dgyzdy(x, z, phi) &
            - this%normb_z(x, z, phi) * this%dgyxdy(x, z, phi) &
            + (this%dgyxdz(x, z, phi) - this%dgyzdx(x, z, phi)) &
        )
    end function

    real(kind=GP) function dabsBdx_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx

        dqdx = this%dqdx(x, z, phi)
        D = this%D(x, z, phi)

        dabsBdx_circular = 1.0_GP / (abs(this%q(x, z, phi)) * D) &
            * (x - this%rho(x, z, phi)**2 / this%q(x, z, phi) * dqdx)
    end function

    real(kind=GP) function dabsBdz_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz

        dqdz= this%dqdz(x, z, phi)
        D = this%D(x, z, phi)

        dabsBdz_circular = 1.0_GP / (abs(this%q(x, z, phi)) * D) &
            * (z - this%rho(x, z, phi)**2 / this%q(x, z, phi) * dqdz)
    end function

    real(kind=GP) function dabsBdy_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dabsBdy_circular = this%normb_x(x, z, phi) * this%dabsBdx(x, z, phi) &
                         + this%normb_z(x, z, phi) * this%dabsBdz(x, z, phi)
    end function

    real(kind=GP) function theta_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        theta_circular = atan2(z, x)
    end function

    real(kind=GP) function jacobian_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        jacobian_circular = 1.0_GP
    end function

    real(kind=GP) function psi_circular(this, x, z, phi)
        class(circular_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        psi_circular = 0.0_GP
    end function

    ! salpha implementations

    subroutine initialize_salpha(this, q_ref, shear, minor_r, L_ref, B_ref)
        class(salpha_reference_t), intent(inout) :: this
        real(kind=GP), intent(in) :: q_ref, shear, minor_r, L_ref, B_ref

        this%q_ref = q_ref
        this%shear = shear
        this%minor_r = minor_r
        this%L_ref = L_ref
        this%B_ref = B_ref
    end subroutine

    real(kind=GP) function rho_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        rho_salpha = sqrt((x - 1.0_GP)**2 + z**2) / this%minor_r
    end function

    real(kind=GP) function q_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        q_salpha = this%q_ref + this%shear * this%rho(x, z, phi)**2
    end function

    real(kind=GP) function D_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        D_salpha = sqrt((x - 1.0_GP)**2 + z**2 + (this%q(x, z, phi)**2))
    end function

    real(kind=GP) function absB_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        absB_salpha = 1.0_GP / (x * this%q(x, z, phi)) &
                      * sqrt((x - 1)**2 + z**2 + this%q(x, z, phi)**2)
    end function

    real(kind=GP) function normb_x_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_x_salpha = -z / this%D(x, z, phi)
    end function

    real(kind=GP) function normb_y_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_y_salpha = this%q(x, z, phi) / this%D(x, z, phi)
    end function

    real(kind=GP) function normb_z_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        normb_z_salpha = (x - 1.0_GP) / this%D(x, z, phi)
    end function

    real(kind=GP) function dqdx_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dqdx_salpha =  2 * this%shear / this%minor_r**2 * (x - 1.0_GP)
    end function

    real(kind=GP) function dqdz_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dqdz_salpha =  2 * this%shear / this%minor_r**2 * z
    end function

    real(kind=GP) function dgyzdx_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx

        D = this%D(x, z, phi)
        dqdx = this%dqdx(x, z, phi)
        dgyzdx_salpha = 1.0_GP / D - (x - 1.0_GP) / D**3 &
                                     * ((x - 1.0_GP) + this%q(x, z, phi) * dqdx)
    end function

    real(kind=GP) function dgyxdz_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz

        dqdz = this%dqdz(x, z, phi)
        D = this%D(x, z, phi)
        dgyxdz_salpha = -1.0_GP / D + z / D**3 * (z + this%q(x, z, phi) * dqdz)
    end function

    real(kind=GP) function dgyxdx_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx

        dqdx = this%dqdx(x, z, phi)
        D = this%D(x, z, phi)
        dgyxdx_salpha =   z * (this%q(x, z, phi) * dqdx + x - 1.0_GP) / D**3
    end function

    real(kind=GP) function dgyzdz_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz

        dqdz = this%dqdz(x, z, phi)
        D = this%D(x, z, phi)
        dgyzdz_salpha =  -(x - 1.0_GP) * (this%q(x, z, phi) * dqdz + z) / D**3
    end function

    real(kind=GP) function dgyxdy_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: dbxdphi

        dbxdphi = -this%normb_y(x, z, phi)

        dgyxdy_salpha = this%normb_x(x, z, phi) * this%dgyxdx(x, z, phi) &
                      + this%normb_z(x, z, phi) * this%dgyxdz(x, z, phi) &
                      + this%normb_y(x, z, phi) * dbxdphi / x
    end function

    real(kind=GP) function dgyzdy_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dgyzdy_salpha = this%normb_x(x, z, phi) * this%dgyzdx(x, z, phi) &
                      + this%normb_z(x, z, phi) * this%dgyzdz(x, z, phi)
    end function

    real(kind=GP) function curl_normb_y_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        curl_normb_y_salpha = 1.0_GP / abs(this%normb_y(x, z, phi)) * ( &
            + this%normb_x(x, z, phi) * this%dgyzdy(x, z, phi) &
            - this%normb_z(x, z, phi) * this%dgyxdy(x, z, phi) &
            + (this%dgyxdz(x, z, phi) - this%dgyzdx(x, z, phi)) &
        )
    end function

    real(kind=GP) function dabsBdx_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdx, dDdx

        dqdx = this%dqdx(x, z, phi)
        D = this%D(x, z, phi)
        dDdx = 1.0_GP / D * ((x - 1.0_GP) + this%q(x, z, phi) * dqdx)

        dabsBdx_salpha = dDdx / (x * this%q(x, z, phi)) &
            - D / (x**2 * this%q(x, z, phi)) &
            - D / (x * this%q(x, z, phi)**2) * dqdx
    end function

    real(kind=GP) function dabsBdz_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: D, dqdz, dDdz

        dqdz= this%dqdz(x, z, phi)
        D = this%D(x, z, phi)
        dDdz = 1.0_GP / D * (z + this%q(x, z, phi) * dqdz)

        dabsBdz_salpha = dDdz / (x * this%q(x, z, phi)) &
            - D / (x * this%q(x, z, phi)**2) * dqdz
    end function

    real(kind=GP) function dabsBdy_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        dabsBdy_salpha = this%normb_x(x, z, phi) * this%dabsBdx(x, z, phi) &
                       + this%normb_z(x, z, phi) * this%dabsBdz(x, z, phi)
    end function

    real(kind=GP) function theta_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        theta_salpha = atan2(z, x - 1.0_GP)
    end function

    real(kind=GP) function jacobian_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi

        jacobian_salpha = x
    end function

    real(kind=GP) function psi_salpha(this, x, z, phi)
        class(salpha_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        real(kind=GP) :: r

        r = sqrt((x - 1.0_GP)**2 + z**2)

        associate(a => this%q_ref, b => this%shear, c => this%minor_r)
        psi_salpha = &
            (- c**2*Log((a*c**2)/(b + a*c**2)) &
             + c**2*Log((a*c**2 + b*r**2)/(a*c**2)))/(2.0_GP*b)
        end associate
    end function

    ! Dommaschk implementations

    real(kind=GP) function absB_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_absB_domm.txt"
    end function

    real(kind=GP) function normb_x_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_normb_x_domm.txt"
    end function

    real(kind=GP) function normb_y_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_normb_y_domm.txt"
    end function

    real(kind=GP) function normb_z_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_normb_z_domm.txt"
    end function

    real(kind=GP) function curl_normb_y_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_curl_normb_y_domm.txt"
    end function

    real(kind=GP) function dabsBdx_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dabsBdx_domm.txt"
    end function

    real(kind=GP) function dabsBdy_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dabsBdy_domm.txt"
    end function

    real(kind=GP) function dabsBdz_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dabsBdz_domm.txt"
    end function

    real(kind=GP) function dgyzdx_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyzdx_domm.txt"
    end function

    real(kind=GP) function dgyxdz_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyxdz_domm.txt"
    end function

    real(kind=GP) function dgyzdz_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyzdz_domm.txt"
    end function

    real(kind=GP) function dgyxdx_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyxdx_domm.txt"
    end function

    real(kind=GP) function dgyxdy_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyxdy_domm.txt"
    end function

    real(kind=GP) function dgyzdy_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        include "dommaschk_reference/equi_ref_dgyzdy_domm.txt"
    end function

    real(kind=GP) function rho_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        ! NOTE: Dommaschk doesn't have an analytic flux surface label
        rho_domm = 2.0_GP
    end function

    real(kind=GP) function theta_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        ! NOTE: Theta in the mesh is not designed for 3D
        theta_domm = atan2(z, x - 1.0_GP)
    end function

    real(kind=GP) function jacobian_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        jacobian_domm = x
    end function

    real(kind=GP) function psi_domm(this, x, z, phi)
        class(dommaschk_reference_t), intent(in) :: this
        real(kind=GP), intent(in) :: x, z, phi
        ! NOTE: Dommaschk doesn't have an analytic formula for flux
        psi_domm = 0.0_GP
    end function

end module
