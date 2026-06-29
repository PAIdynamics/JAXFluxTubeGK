module params_bsg_m
    !! Module handling the parameters for Block-Structured Grids (BSG) approach
    use genex_fortran_env_m, only: GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use genex_error_handling_m, only: handle_error
    use bsg_types_m, only: MAX_BLOCKS

    implicit none
    private

    integer :: num_blocks = 2
    !! Number of blocks used in BSG
    real(kind=GP), dimension(MAX_BLOCKS) :: vplen_markers = 0.0_GP
    !! Vp length for each block
    real(kind=GP), dimension(MAX_BLOCKS) :: radial_markers = 0.0_GP
    !! Radial locations of the block boundaries
    integer :: bsg_interp_order = 4
    !! Order of interpolation

    namelist / params_bsg / &
        num_blocks, vplen_markers, radial_markers, bsg_interp_order

    public :: read_params_bsg
    public :: write_params_bsg

    public :: get_num_bsg_blocks
    public :: get_vplen_markers
    public :: get_radial_markers
    public :: get_bsg_interp_order

contains

    integer function get_num_bsg_blocks()
        !! Returns the number of blocks used in BSG
        call handle_bounds_error(num_blocks, 1, MAX_BLOCKS, &
                                 __LINE__, __FILE__)
        get_num_bsg_blocks = num_blocks
    end function

    function get_vplen_markers() result(arr)
        !! Returns the length of the vp space for each block
        real(kind=GP), dimension(MAX_BLOCKS) :: arr
        !! Array of vp lengths for each block

        arr = vplen_markers
    end function

    function get_radial_markers() result(arr)
        !! Returns the radial locations of the block boundaries
        real(kind=GP), dimension(MAX_BLOCKS) :: arr
        !! Radial locations of the block boundaries

        arr = radial_markers
    end function

    integer function get_bsg_interp_order()
        !! Returns the order of interpolation
        call handle_bounds_error(bsg_interp_order, 2, 5, &
                                 __LINE__, __FILE__)
        get_bsg_interp_order = bsg_interp_order
    end function

    subroutine write_params_bsg(filename)
        !! Writes the BSG namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_bsg, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="bsg")

        close(iunit)
    end subroutine

    subroutine read_params_bsg(filename)
        !! Reads the BSG namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_bsg, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="bsg", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
