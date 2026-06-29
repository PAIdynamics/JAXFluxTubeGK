module params_normalization_m
    !! Module handling the parameters for the normalization
    use genex_fortran_env_m, only: GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, get_print_info
    use logger_m, only: logger_get_debug_channel
    use interop_params_m, only: set_cxx_params_normalization

    implicit none
    private

    ! Normalization parameters which can be set in the parameter file
    real(kind=GP) :: m_ref = 1.0_GP
    !! Reference mass (in proton mass)
    real(kind=GP) :: T_ref = 1.0_GP
    !! Reference temperature (in keV)
    real(kind=GP) :: n_ref = 1.0_GP
    !! Reference density (in 1e19 m^{-3})
    real(kind=GP) :: L_ref = 1.0_GP
    !! Reference length (in m)
    real(kind=GP) :: B_ref = 1.0_GP
    !! Reference magnetic field (in T)
    real(kind=GP) :: P_ref = 1.0_GP
    !! Reference power (in kW)

    ! Derived normalization parameters
    real(kind=GP) :: c_ref = 3.09496901846338e5_GP
    !! Reference velocity (in m/s)
    real(kind=GP) :: rho_ref = 3.23105013899193e-3_GP
    !! Reference gyroradius (in m)
    real(kind=GP) :: coll_ref = 2.7719928e-4_GP
    !! Reference collision frequency prefactor (unitless)
    !! NOTE: Not related to nu_ref or collisionality
    real(kind=GP) :: beta_ref = 403.0e-5_GP
    !! Reference beta (unitless)
    real(kind=GP) :: diff_ref = 3.09496901846338e5_GP
    !! Reference neutrals diffusion coefficient (in m^2/s)
    real(kind=GP) :: rate_ref = 3.09496901846338e5_GP
    !! Reference reaction rate coefficient (in 1e-19 m^3/s)
    real(kind=GP) :: xsec_ref = 1.0_GP
    !! Reference cross-section coefficient (in 1e-19 m^2)
    real(kind=GP) :: heat_ref = 4.958687044335941e5_GP
    !! Reference heat source coefficient (in kW)
    real(kind=GP) :: fuel_ref = 3.09496901846338e5_GP
    !! Reference fuelling (particle source) coefficient (in 1e19/s)
    real(kind=GP) :: torque_ref = 1.602176634_GP
    !! Reference torque (momentum) density injection coefficient (in kN/m^3)

    namelist / params_normalization / m_ref, T_ref, n_ref, L_ref, B_ref, P_ref

    public :: read_params_normalization
    public :: write_params_normalization

    public :: get_m_ref
    public :: get_T_ref
    public :: get_n_ref
    public :: get_L_ref
    public :: get_B_ref
    public :: get_P_ref

    public :: get_c_ref
    public :: get_rho_ref
    public :: get_coll_ref
    public :: get_beta_ref
    public :: get_diff_ref
    public :: get_rate_ref
    public :: get_xsec_ref
    public :: get_heat_ref
    public :: get_fuel_ref
    public :: get_torque_ref

contains

    pure real(kind=GP) function get_m_ref()
        !! Returns the reference mass
        get_m_ref = m_ref
    end function

    pure real(kind=GP) function get_T_ref()
        !! Returns the reference temperature
        get_T_ref = T_ref
    end function

    pure real(kind=GP) function get_n_ref()
        !! Returns the reference density
        get_n_ref = n_ref
    end function

    pure real(kind=GP) function get_L_ref()
        !! Returns the reference length
        get_L_ref = L_ref
    end function

    pure real(kind=GP) function get_B_ref()
        !! Returns the reference magnetic field
        get_B_ref = B_ref
    end function

    pure real(kind=GP) function get_P_ref()
        !! Returns the reference power
        get_P_ref = P_ref
    end function

    pure real(kind=GP) function get_rho_ref()
        !! Returns the reference gyroradius
        get_rho_ref = rho_ref
    end function

    pure real(kind=GP) function get_c_ref()
        !! Returns the reference velocity
        get_c_ref = c_ref
    end function

    pure real(kind=GP) function get_coll_ref()
        !! Returns the reference collision frequency prefactor
        get_coll_ref = coll_ref
    end function

    pure real(kind=GP) function get_beta_ref()
        !! Returns the reference beta
        get_beta_ref = beta_ref
    end function

    pure real(kind=GP) function get_diff_ref()
        !! Returns the reference diffusion coefficient
        get_diff_ref = diff_ref
    end function

    pure real(kind=GP) function get_rate_ref()
        !! Returns the reference reaction rate coefficient
        get_rate_ref = rate_ref
    end function

    pure real(kind=GP) function get_xsec_ref()
        !! Returns the reference cross-section coefficient
        get_xsec_ref = xsec_ref
    end function

    pure real(kind=GP) function get_heat_ref()
        !! Returns the reference heat source coefficient
        get_heat_ref = heat_ref
    end function

    pure real(kind=GP) function get_fuel_ref()
        !! Returns the reference particle fuelling coefficient
        get_fuel_ref = fuel_ref
    end function

    pure real(kind=GP) function get_torque_ref()
        !! Returns the reference torque injection coefficient
        get_torque_ref = torque_ref
    end function

    subroutine calc_rho_ref()
        !! Calculates the reference gyroradius
        call calc_c_ref()
        rho_ref = c_ref * m_ref / B_ref * 1.043968492e-8_GP
    end subroutine

    subroutine calc_c_ref()
        !! Calculates the reference velocity
        c_ref = 9.787151386e3_GP * sqrt((T_ref * 1e3_GP) / m_ref)
    end subroutine

    subroutine calc_coll_ref()
        !! Calculates the reference collision frequency prefactor
        coll_ref = 2.7719928e-4_GP * L_ref * n_ref / (T_ref**2)
    end subroutine

    subroutine calc_beta_ref()
        !! Calculates the reference beta
        beta_ref = 403.0e-5_GP * n_ref * T_ref / (B_ref * B_ref)
    end subroutine

    subroutine calc_diff_ref()
        !! Calculates the reference diffusion coefficient
        call calc_c_ref()
        diff_ref = L_ref * c_ref
    end subroutine

    subroutine calc_rate_ref()
        !! Calculates the reference reaction rate
        call calc_c_ref()
        rate_ref = c_ref / (L_ref * n_ref)
    end subroutine

    subroutine calc_xsec_ref()
        !! Calculates the reference cross-section rate
        xsec_ref = 1.0_GP / (L_ref * n_ref)
    end subroutine

    subroutine calc_fuel_ref()
        !! Calculates the reference fuelling coefficient
        call calc_c_ref()
        fuel_ref = L_ref**2 * c_ref * n_ref
    end subroutine

    subroutine calc_torque_ref()
        !! Calculates the reference torque injection coefficient
        torque_ref = n_ref * T_ref / L_ref &
                   * 1.602176634_GP
    end subroutine

    subroutine calc_heat_ref()
        call calc_c_ref()
        call calc_fuel_ref()
        !! Calculates the reference heat source coefficient
        heat_ref = 1.602176634_GP * T_ref * fuel_ref
    end subroutine

    subroutine write_params_normalization(filename)
        !! Writes the normalization namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_normalization, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="normalization")

        close(iunit)
    end subroutine

    subroutine read_params_normalization(filename)
        !! Reads the normalization namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_normalization, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="normalization", &
                               iunit=iunit)

        close(iunit)

        ! Initialize derived reference quantities with new parameters
        call calc_c_ref()
        call calc_rho_ref()
        call calc_coll_ref()
        call calc_beta_ref()
        call calc_diff_ref()
        call calc_rate_ref()
        call calc_xsec_ref()
        call calc_fuel_ref()
        call calc_torque_ref()
        call calc_heat_ref()

        if(get_print_info()) then
            write(logger_get_debug_channel(), *) &
                    "Calculating derived reference quantities:"
            write(logger_get_debug_channel(), *) "  c_ref:    ", c_ref
            write(logger_get_debug_channel(), *) "  rho_ref:  ", rho_ref
            write(logger_get_debug_channel(), *) "  coll_ref: ", coll_ref
            write(logger_get_debug_channel(), *) "  beta_ref: ", beta_ref
            write(logger_get_debug_channel(), *) "  diff_ref: ", diff_ref
            write(logger_get_debug_channel(), *) "  rate_ref: ", rate_ref
            write(logger_get_debug_channel(), *) "  xsec_ref: ", xsec_ref
            write(logger_get_debug_channel(), *) "  fuel_ref: ", fuel_ref
            write(logger_get_debug_channel(), *) "  torque_ref: ", torque_ref
            write(logger_get_debug_channel(), *) "  heat_ref: ", heat_ref
        endif

        call set_cxx_params_normalization()
    end subroutine

end module params_normalization_m
