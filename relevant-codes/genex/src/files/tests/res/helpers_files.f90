module helpers_files_m
    !! Contains procedures needed for unit testing the files types
    use mpi
    use netcdf
    use genex_fortran_env_m, only: GP, GP_EPS
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_UTESTS
    use math_m, only: almost_equal

    implicit none
    private

    public :: choose_mpi_ranks
    public :: check_dim_records
    public :: check_global_records
    public :: check_val_records

    interface check_global_records
        ! Checks the value of global variables
        module procedure check_global_records_int
        module procedure check_global_records_real
    end interface

    interface check_val_records
        ! Checks the values of a variable chosen by name.
        module procedure check_val_records_1dreal
        module procedure check_val_records_2dreal
        module procedure check_val_records_3dreal
        module procedure check_val_records_1dinteger
        module procedure check_val_records_2dinteger
    end interface

contains

    subroutine choose_mpi_ranks(n_ranks, n_ranks_phi, n_ranks_vp, n_ranks_mu, &
                                n_ranks_sp)
        !! Chooses mpi ranks depending on the total ranks to test at least
        !! all dim parallelizations
        integer, intent(in) :: n_ranks
        integer, intent(out) :: n_ranks_phi, n_ranks_vp, n_ranks_mu, n_ranks_sp

        select case (n_ranks)
        case (1)
          n_ranks_phi = 1
          n_ranks_vp = 1
          n_ranks_mu = 1
          n_ranks_sp = 1
        case (2)
          n_ranks_phi = 2
          n_ranks_vp = 1
          n_ranks_mu = 1
          n_ranks_sp = 1
        case (4)
          n_ranks_phi = 1
          n_ranks_vp = 2
          n_ranks_mu = 2
          n_ranks_sp = 1
        case (8)
          n_ranks_phi = 1
          n_ranks_vp = 2
          n_ranks_mu = 2
          n_ranks_sp = 2
        case default
            call handle_error("Unsupported n_procs in unit test!", &
                              GENEX_ERR_UTESTS, __LINE__, __FILE__)
        end select
    end subroutine

    function check_global_records_int(filename, globalname, expected, &
                is_master, groupname) result(res)
        ! Checks the value of the integer NetCDF global variable given by name.
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: globalname
        integer, intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, read_id
        integer :: global

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get global value
            ierr = nf90_get_att(read_id, NF90_GLOBAL, globalname, global)
            ! Check
            if(global /= expected) res = .false.
        endif
    end function

    function check_global_records_real(filename, globalname, expected, &
                is_master, groupname) result(res)
        ! Checks the value of the real NetCDF global variable given by name.
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: globalname
        real(kind=GP), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, read_id
        real(kind=GP) :: global

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get global value
            ierr = nf90_get_att(read_id, NF90_GLOBAL, globalname, global)
            ! Check
            if(.not. almost_equal(global, expected, GP_EPS)) res = .false.
        endif
    end function

    function check_dim_records(filename, dimname, expected, is_master, &
                groupname) result(res)
        ! Checks the dimensions chosen by name.
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: dimname
        integer, intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, dim_id, read_id, n_records

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get dimension id
            ierr = nf90_inq_dimid(read_id, dimname, dim_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read dimension
            ierr = nf90_inquire_dimension(read_id, dim_id, len = n_records)
            if(ierr /= NF90_NOERR) res = .false.
            ! Check
            if(n_records /= expected) res = .false.
        endif
    end function

    function check_val_records_1dreal(filename, varname, expected, is_master, &
                groupname) result(res)
        ! Implementation for 1d real arrays
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: varname
        real(kind = GP), dimension(:), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, read_id, var_id, i
        real(kind = GP), dimension(size(expected)) :: values

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get var id
            ierr = nf90_inq_varid(read_id, varname, var_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read var
            ierr = nf90_get_var(read_id, var_id, values)
            !Check
            do i = 1, size(values)
                if(.not.almost_equal(values(i), expected(i), GP_EPS)) then
                    res = .false.
                    exit
                end if
            end do
        endif
    end function

    function check_val_records_2dreal(filename, varname, expected, is_master, &
                groupname, time) result(res)
        ! Implementation for 2d real arrays
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: varname
        real(kind = GP), dimension(:, :), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        integer, intent(in), optional :: time
        logical :: res

        integer :: ierr, file_id, read_id, var_id, i, j, start(3)
        real(kind = GP), dimension(size(expected, 1),size(expected, 2)) ::values

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get var id
            ierr = nf90_inq_varid(read_id, varname, var_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read var
            if(present(time)) then
                start = [1, 1, time]
                ierr = nf90_get_var(read_id, var_id, values, start = start)
            else
                ierr = nf90_get_var(read_id, var_id, values)
            endif
            !Check
            do i = 1, size(values, 1)
                do j = 1, size(values, 2)
                    if(.not.almost_equal(values(i, j), &
                                         expected(i, j), GP_EPS)) then
                        res = .false.
                        exit
                    end if
                end do
            end do
        endif
    end function

    function check_val_records_3dreal(filename, varname, expected, is_master, &
                groupname, time) result(res)
        ! Implementation for 3d real arrays
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: varname
        real(kind = GP), dimension(:, :, :), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        integer, intent(in), optional :: time
        logical :: res

        integer :: ierr, file_id, read_id, var_id, i, j, k, start(3)
        real(kind = GP), dimension(size(expected, 1),size(expected, 2), &
                                   size(expected, 3)) :: values

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get var id
            ierr = nf90_inq_varid(read_id, varname, var_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read var
            if(present(time)) then
                start = [1, 1, time]
                ierr = nf90_get_var(read_id, var_id, values, start = start)
            else
                ierr = nf90_get_var(read_id, var_id, values)
            endif
            !Check
            do i = 1, size(values, 1)
                do j = 1, size(values, 2)
                    do k = 1, size(values, 3)
                        if(.not.almost_equal(values(i, j, k), &
                                             expected(i, j, k), GP_EPS)) then
                            res = .false.
                            exit
                        end if
                    end do
                end do
            end do
        endif
    end function

    function check_val_records_1dinteger(filename, varname, expected, &
                is_master, groupname) result(res)
        ! Implementation for 1d integer arrays
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: varname
        integer, dimension(:), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, read_id, var_id, i
        integer :: values(size(expected))

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get var id
            ierr = nf90_inq_varid(read_id, varname, var_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read var
            ierr = nf90_get_var(read_id, var_id, values)
            !Check
            do i = 1, size(values)
                if(values(i) /= expected(i)) then
                    res = .false.
                    exit
                end if
            end do
        endif
    end function

    function check_val_records_2dinteger(filename, varname, expected, &
                is_master, groupname) result(res)
        ! Implementation for 2d integer arrays
        character(len = *), intent(in) :: filename
        character(len = *), intent(in) :: varname
        integer, dimension(:, :), intent(in) :: expected
        logical, intent(in) :: is_master
        character(len = *), intent(in), optional :: groupname
        logical :: res

        integer :: ierr, file_id, read_id, var_id, i, j
        integer, dimension(size(expected, 1), size(expected, 2)) :: values

        res = .true.
        if(is_master) then
            ! Open file
            ierr = nf90_open(filename, NF90_NOWRITE, file_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Get group id if specified
            if(present(groupname)) then
                ierr = nf90_inq_ncid(file_id, groupname, read_id)
                if(ierr /= NF90_NOERR) res = .false.
            else
                read_id = file_id
            end if
            ! Get var id
            ierr = nf90_inq_varid(read_id, varname, var_id)
            if(ierr /= NF90_NOERR) res = .false.
            ! Read var
            ierr = nf90_get_var(read_id, var_id, values)
            !Check
            do i = 1, size(values, 1)
                do j = 1, size(values, 2)
                    if(values(i, j) /= expected(i, j)) then
                        res = .false.
                        exit
                    end if
                end do
            end do
        endif
    end function
end module
