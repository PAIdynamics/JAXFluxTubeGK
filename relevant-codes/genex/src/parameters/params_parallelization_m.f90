module params_parallelization_m
    !! Module handling the parameters for the parallelization
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS

    implicit none
    private

    integer :: n_procs_RZ = 1
    !! Number of MPI processes in RZ direction
    integer :: n_procs_vp = 1
    !! Number of MPI processes in VP direction
    integer :: n_procs_mu = 1
    !! Number of MPI processes in MU direction
    integer :: n_procs_phi = 1
    !! Number of MPI processes in PHI direction
    integer :: n_procs_sp = 1
    !! Number of MPI processes in SP direction

    namelist / params_parallelization / &
        n_procs_RZ, n_procs_vp, n_procs_mu, n_procs_phi, n_procs_sp

    public :: read_params_parallelization
    public :: write_params_parallelization

    public :: get_n_procs_RZ
    public :: get_n_procs_vp
    public :: get_n_procs_mu
    public :: get_n_procs_phi
    public :: get_n_procs_sp

contains

    pure integer function get_n_procs_RZ()
        !! Returns the number of processes in R direction
        get_n_procs_RZ = n_procs_RZ
    end function

    pure integer function get_n_procs_vp()
        !! Returns the number of processes in vp direction
        get_n_procs_vp = n_procs_vp
    end function

    pure integer function get_n_procs_mu()
        !! Returns the number of processes in mu direction
        get_n_procs_mu = n_procs_mu
    end function

    pure integer function get_n_procs_phi()
        !! Returns the number of processes in phi direction
        get_n_procs_phi = n_procs_phi
    end function

    pure integer function get_n_procs_sp()
        !! Returns the number of processes in sp direction
        get_n_procs_sp = n_procs_sp
    end function

    subroutine write_params_parallelization(filename)
        !! Writes the parallelization namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_parallelization, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="parallelization")

        close(iunit)
    end subroutine

    subroutine read_params_parallelization(filename)
        !! Reads the parallelization namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_parallelization, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="parallelization", &
                               iunit=iunit)

        close(iunit)

        call check_n_procs(n_procs_RZ, "RZ")
        call check_n_procs(n_procs_phi, "phi")
        call check_n_procs(n_procs_vp, "vp")
        call check_n_procs(n_procs_mu, "mu")
        call check_n_procs(n_procs_sp, "sp")

    contains

        subroutine check_n_procs(n_procs, proc_name)
            integer, intent(in) :: n_procs
            character(len=*), intent(in) :: proc_name

            if (n_procs < 1) then
                call handle_error("Number of "//trim(proc_name) &
                                  //" processes was < 1!", &
                                  GENEX_ERR_PARAMETERS, __LINE__, __FILE__)
            endif
        end subroutine

    end subroutine

end module params_parallelization_m
