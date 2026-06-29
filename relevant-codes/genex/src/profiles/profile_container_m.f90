module profile_container_m
    !! Contains a type that contains a collection of profiles for one species
    use genex_fortran_env_m, only : GP
    use profiles_m, only: profile_t
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_PARAMETERS

    implicit none
    private

    type, public :: profile_container_t
        !! Type that contains a collection of profiles for one species
        logical, private, allocatable :: is_isotropic
        !! Indicates if this container has isotropic or anisotropic profiles
        class(profile_t), private, allocatable :: dens
        !! Density profile
        class(profile_t), private, allocatable :: temp
        !! Temperature profile
        class(profile_t), private, allocatable :: temp_par
        !! Parallel temperature profiles
        class(profile_t), private, allocatable :: temp_perp
        !! Perpendicular temperature profiles
    contains
        procedure, public :: initialize
        !! Initializes the profile container
        procedure, public :: get_is_isotropic
        !! Get a logical that indicates if the species is isotropic
        procedure, public :: eval_dens
        !! Evaluates the density for given real space coordinates
        procedure, public :: eval_temp
        !! Evaluates the temperature for given real space coordinates
        procedure, public :: eval_temp_par
        !! Evaluates the parallel temperature for given real space coordinates
        procedure, public :: eval_temp_perp
        !! Evaluates the perpendicular temperature for given real space coords
    end type

contains
    subroutine initialize(this, is_isotropic, dens, temp, temp_par, temp_perp)
        class(profile_container_t), intent(inout) :: this
        !! Instance of the type
        logical, intent(in) :: is_isotropic
        !! Indicates if this container has isotropic or anisotropic profiles
        class(profile_t), intent(in) :: dens
        !! Density profile
        class(profile_t), intent(in) :: temp
        !! Temperature profile
        class(profile_t), intent(in) :: temp_par
        !! Parallel temperature profile
        class(profile_t), intent(in) :: temp_perp
        !! Perpendicular temperature profile

        this%is_isotropic = is_isotropic
        this%dens         = dens
        this%temp         = temp
        this%temp_par     = temp_par
        this%temp_perp    = temp_perp
    end subroutine

    logical pure function get_is_isotropic(this)
        class(profile_container_t), intent(in) :: this
        !! Instance of the type

        get_is_isotropic = this%is_isotropic
    end function

    subroutine eval_dens(this, rho, theta, phi, res, ierr)
        class(profile_container_t), intent(in) :: this
        !! Instance of the type
        real(kind = GP), intent(in) :: rho
        !! Normalized poloidal flux surface label
        real(kind = GP), intent(in) :: theta
        !! Poloidal angle
        real(kind = GP), intent(in) :: phi
        !! Toroidal angle
        real(kind = GP), intent(out) :: res
        !! Result
        integer, intent(out) :: ierr
        !! Error code

        res = this%dens%eval(rho, theta, phi)
        ierr = GENEX_SUCCESS
    end subroutine

    subroutine eval_temp(this, rho, theta, phi, res, ierr)
        class(profile_container_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi
        real(kind = GP), intent(out) :: res
        integer, intent(out) :: ierr

        if(this%is_isotropic) then
            res = this%temp%eval(rho, theta, phi)
            ierr = GENEX_SUCCESS
        else
            ierr = GENEX_ERR_PARAMETERS
        end if
    end subroutine

    subroutine eval_temp_par(this, rho, theta, phi, res, ierr)
        class(profile_container_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi
        real(kind = GP), intent(out) :: res
        integer, intent(out) :: ierr

        if(.not. this%is_isotropic) then
            res = this%temp_par%eval(rho, theta, phi)
            ierr = GENEX_SUCCESS
        else
            ierr = GENEX_ERR_PARAMETERS
        end if
    end subroutine

    subroutine eval_temp_perp(this, rho, theta, phi, res, ierr)
        class(profile_container_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi
        real(kind = GP), intent(out) :: res
        integer, intent(out) :: ierr

        if(.not. this%is_isotropic) then
            res = this%temp_perp%eval(rho, theta, phi)
            ierr = GENEX_SUCCESS
        else
            ierr = GENEX_ERR_PARAMETERS
        end if
    end subroutine

end module
