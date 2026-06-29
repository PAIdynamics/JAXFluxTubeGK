module params_dist_initial_slowing_down_m
    !! Module handling the parameters for dist initial slowing down
    use genex_fortran_env_m, only: GP
    use params_dist_initial_m, only: params_dist_initial_t
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_dist_initial_t) :: &
                                             params_dist_initial_slowing_down_t
        !! Container type to store params for the slowing down dist init
        real(kind=GP), public :: vbirth  = 1.0_GP
        !! Birth speed in units of the species thermal speed
    end type

    type(params_dist_initial_slowing_down_t), dimension(n_spec_supported) :: &
                                        params_dist_initial_slowing_down_array

    public :: read_params_dist_initial_slowing_down
    public :: write_params_dist_initial_slowing_down

    public :: get_params_dist_initial_slowing_down

contains

    function get_params_dist_initial_slowing_down(n) result(par)
        integer, intent(in) :: n
        type(params_dist_initial_slowing_down_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_dist_initial_slowing_down_array(n)
    end function

    subroutine write_params_dist_initial_slowing_down(filename, n)
        !! Writes the slowing_down namelist to the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: vbirth

        namelist / params_dist_initial_slowing_down / vbirth

        vbirth  = params_dist_initial_slowing_down_array(n)%vbirth

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_dist_initial_slowing_down, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="dist_initial_slowing_down")

        close(iunit)
    end subroutine

    subroutine read_params_dist_initial_slowing_down(filename, n)
        !! Reads the slowing_down namelist from the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: vbirth

        namelist / params_dist_initial_slowing_down / vbirth

        ! Initialize temporaries with default values from params array
        vbirth  = params_dist_initial_slowing_down_array(n)%vbirth

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_dist_initial_slowing_down, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="dist_initial_slowing_down", &
                               iunit=iunit)

        params_dist_initial_slowing_down_array(n)%vbirth  = vbirth

        close(iunit)
    end subroutine

end module
