module params_neutrals_config_m
    !! Module handling the configurations for the different neutrals species.
    use genex_fortran_env_m, only : GP, GP_EPS
    use params_neutrals_m, only: n_neut_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    ! Configuration for all the neutral species (model used)
    character(len=16) :: neutrals_evolve_type = "none"
    !! Neutrals evolution operator to be used.
    !! Current options are: "none", "dummy", "1mom_2cfd", "1mom_4cfd"
    character(len=16) :: neutrals_coupling_type = "none"
    !! Neutrals coupling operator to be used.
    !! Current options are: "none", "dummy", "krook"
    real(kind=GP) :: neutrals_dens_floor = 1.0e-6_GP
    !! Parameter that sets a minimum limit on the density in the
    !! calculations of plasma moments used for the neutral coupling.
    ! NOTE: The density floor is set with respect to n_ref = 1e19 m^{-3}.
    !       If n_ref is changed in params_normalization_m, neutrals_dens_floor
    !       needs to be adjusted accordingly to remain equal to 1e13 m^{-3}.
    real(kind=GP) :: neutrals_temp_floor = 1.0e-3_GP
    !! Parameter that sets a minimum limit on the temperature in the
    !! calculations of plasma to neutral coupling.
    ! NOTE: The temperature floor is set with respect to T_ref = 1 keV.
    !       If T_ref is changed in params_normalization_m, neutrals_temp_floor
    !       needs to be adjusted accordingly to remain equal to 1eV.

    ! Configuration for Krook coupling operator
    real(kind=GP) :: neutrals_gamma_u = 1.0_GP
    !! Set the neutrals parallel velocity to neutrals_gamma_u * v_ions
    real(kind=GP) :: neutrals_gamma_T = 1.0_GP
    !! Set the neutrals parallel temperature to neutrals_gamma_T * T_ions
    real(kind=GP) :: neutrals_gamma_W = 0.5_GP
    !! Set the electrons cooling temperature due to neutrals to
    !! neutrals_gamma_W * T_elec

    namelist / params_neutrals_config / neutrals_evolve_type, &
                                        neutrals_coupling_type, &
                                        neutrals_dens_floor, &
                                        neutrals_temp_floor, &
                                        neutrals_gamma_u, &
                                        neutrals_gamma_T, &
                                        neutrals_gamma_W

    public :: get_use_neutrals

    public :: get_neut_evolve_type
    public :: get_neut_coupling_type
    public :: get_neut_temp_floor
    public :: get_neut_dens_floor
    public :: get_neut_gamma_u
    public :: get_neut_gamma_T
    public :: get_neut_gamma_W

    public :: read_params_neutrals_config
    public :: write_params_neutrals_config

contains
    pure function get_use_neutrals() result(res)
        !! Returns true if neutrals evolve and coupling are not none
        logical :: res
        res = .true.
        if (neutrals_coupling_type == "none" &
                .or. neutrals_evolve_type == "none") then
            res = .false.
        end if
    end function

    pure function get_neut_evolve_type() result(res)
        !! Returns neutrals evolution type of the simulation
        character(len=16) :: res
        res = neutrals_evolve_type
    end function

    pure function get_neut_coupling_type() result(res)
        !! Returns neutrals coupling type of the simulation
        character(len=16) :: res
        res = neutrals_coupling_type
    end function

    pure function get_neut_dens_floor() result(res)
        !! Returns the floor value for the density
        real(kind=GP) :: res
        res = neutrals_dens_floor
    end function

    pure function get_neut_temp_floor() result(res)
        !! Returns the floor value for the temperature
        real(kind=GP) :: res
        res = neutrals_temp_floor
    end function

    pure function get_neut_gamma_u() result(res)
        !! Returns the gamma_u factor for the Krook neutrals coupling operator
        real(kind=GP) :: res
        res = neutrals_gamma_u
    end function

    pure function get_neut_gamma_T() result(res)
        !! Returns the gamma_T factor for the Krook neutrals coupling operator
        real(kind=GP) :: res
        res = neutrals_gamma_T
    end function

    pure function get_neut_gamma_W() result(res)
        !! Returns the gamma_W factor for the Krook neutrals coupling operator
        real(kind=GP) :: res
        res = neutrals_gamma_W
    end function

    subroutine write_params_neutrals_config(filename)
        !! Writes the neutrals config namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_neutrals_config, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="neutrals_config")

        close(iunit)
    end subroutine

    subroutine read_params_neutrals_config(filename)
        !! Reads the neutrals config namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_neutrals_config, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="neutrals_config", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
