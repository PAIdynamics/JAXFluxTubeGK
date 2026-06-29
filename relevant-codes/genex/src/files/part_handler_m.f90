module part_handler_m
    !! Module that handles the structuring of the output into different
    !! parts. The parts are labelled by consecutive numbers in the diagnostic
    !! directory as:
    !! part_1
    !! part_2
    !! ...
    !! part_9999
    !! Currently a maximum of 9999 parts is supported.
    use dcomm_handler_m, only: dcomm_handler_t
    use type_converters_m, only: string
    use system_calls_m, only: mkdir
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_DIAG_FILES
    use mpi
    implicit none
    private

    public :: inquire_part, create_part

contains

    subroutine search_part(diag_dir, current_path, next_path)
        !! Searches in the diagnostic directory for the part directory with the
        !! highest number. If current_part is present it contains the part
        !! directory with the highest number found. If no part directory exists
        !! it contains an empty string. If next_part is present it contains
        !! an empty string if "part_9999" exist, it contains "part_1" if
        !! no part directory exist. Else it contains the part directory with
        !! one number higher than the highest number found.
        character(len=*), intent(in) :: diag_dir
        !! Path to the diagnostic directory
        character(len=:), allocatable, optional, intent(out) :: current_path
        !! Path to the current part directory
        character(len=:), allocatable, optional, intent(out) :: next_path
        !! Path to the next part directory

        integer, parameter :: max_search = 1e4 - 1
        integer :: part_id
        logical :: exists
        character(len=:), allocatable :: filename, current_path_loc, &
                                         next_path_loc
        character(len=4) :: part_id_str

        exists = .false.
        do part_id = max_search, 1, -1
            current_path_loc = trim(diag_dir) // "part_" &
                                              // string(part_id) // "/"
            if(part_id == max_search) then
                next_path_loc = ""
            else
                next_path_loc = trim(diag_dir) // "part_" &
                                               // string(part_id + 1) // "/"
            endif
            !! In standard Fortran it is not possible to inquire the existence
            !! of a directory. Therefore, we decide whether a part folder
            !! exists by inquiring the existence of the mom_0d.nc file in the
            !! part folder.
            filename = current_path_loc // "mom_0d.nc"
            inquire(file=filename, exist=exists)
            if(exists) exit
        enddo
        if(.not. exists) then
            current_path_loc = ""
            next_path_loc = trim(diag_dir) // "part_" // string(1) // "/"
        endif
        if(present(current_path)) then
            current_path = current_path_loc
        endif
        if(present(next_path)) then
            next_path = next_path_loc
        endif
    end subroutine

    function inquire_part(dcomm_handler, diag_dir) result(part_path)
        !! Returns the path to the latest part directory. If no part directory
        !! exists an empty string is returned.
        type(dcomm_handler_t), intent(in) :: dcomm_handler
        !! Pointer to the dcomm_handler
        character(len=*), intent(in) :: diag_dir
        !! Path to the diagnostic directory
        character(len=:), allocatable :: part_path
        integer :: ierr

        call search_part(diag_dir, current_path = part_path)
        call mpi_barrier(dcomm_handler%get_comm_cart(), ierr)
    end function

    function create_part(dcomm_handler, diag_dir) result(part_path)
        !! Creates and returns the path to a new part directory. If no part
        !! directory exists "part_1" is returned. I the number of parts
        !! is exhausted and the directory "part_9999" is present an empty
        !! string is returned.
        type(dcomm_handler_t), intent(in) :: dcomm_handler
        !! Pointer to the dcomm_handler
        character(len=*), intent(in) :: diag_dir
        !! Path to the diagnostic directory
        character(len=:), allocatable :: part_path
        integer :: ierr

        call search_part(diag_dir, next_path = part_path)
        call mpi_barrier(dcomm_handler%get_comm_cart(), ierr)
        if(part_path /= "" .and. dcomm_handler%is_master()) then
            call mkdir(part_path, ierr)
        endif
        if(ierr /= GENEX_SUCCESS) then
            call handle_error("Failed to create part in """//diag_dir//"""!", &
                              GENEX_ERR_DIAG_FILES, __LINE__, __FILE__)
        endif
        call mpi_barrier(dcomm_handler%get_comm_cart(), ierr)
    end function
end module
