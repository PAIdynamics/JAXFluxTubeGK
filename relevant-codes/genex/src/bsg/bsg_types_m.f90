module bsg_types_m
    !! Module defining different BSG types
    use genex_fortran_env_m, only: GP
    use genex_status_codes_m, only: GENEX_ERR_BSG
    use genex_error_handling_m, only: handle_error, error_info_t

    implicit none

    private

    integer, public, parameter :: MAX_BLOCKS = 15

    ! Constants for bit positions
    integer, parameter :: BFLAG_BITS = 4
    integer, parameter :: FLAG_P_BITS = 1
    integer, parameter :: FLAG_PP_BITS = 1
    integer, parameter :: FLAG_M_BITS = 1
    integer, parameter :: FLAG_MM_BITS = 1

    ! Pack and unpack routines
    public :: pack_flags
    public :: unpack_bflag
    public :: unpack_all_flags

    type, public :: vspace_bsg_t
        !! Derived type to access velocity space grids for BSG. This additional
        !! layer is added to make sure that the velocity space grids are not
        !! directly accessed in the OpenMP parallel region through getters.
        real(kind=GP), dimension(:), pointer :: array
        !! Pointer to the array containing the velocity space grids
    end type vspace_bsg_t

contains

    subroutine pack_flags(bflag, flag_p, flag_pp, flag_m, flag_mm, packed_flag)
        !! Subroutine to pack flags: bflag, flag_p, flag_pp, flag_m, flag_mm
        !! 1.) Flag for block number (1 to MAX_BLOCKS): bflag
        !!     Bits needed: 4 bits (2^4 = 16 possible values)
        !! 2.) Flag to indicate if k+1 point stencil contains
        !!     points in different blocks (0 or 1): flag_p
        !!     Bits needed: 1 bit (2^1 = 2 possible values)
        !! 3.) Flag to indicate if k+2 point stencil contains
        !!     points in different blocks (0 or 1): flag_pp
        !!     Bits needed: 1 bit (2^1 = 2 possible values)
        !! 4.) Flag to indicate if k-1 point stencil contains
        !!     points in different blocks (0 or 1): flag_m
        !!     Bits needed: 1 bit (2^1 = 2 possible values)
        !! 5.) Flag to indicate if k-2 point stencil contains
        !!     points in different blocks (0 or 1): flag_mm
        !!     Bits needed: 1 bit (2^1 = 2 possible values)

        integer, intent(in) :: bflag
        !! Block number flag
        integer, intent(in) :: flag_p
        !! Flag for k+1 point stencil
        integer, intent(in) :: flag_pp
        !! Flag for k+2 point stencil
        integer, intent(in) :: flag_m
        !! Flag for k-1 point stencil
        integer, intent(in) :: flag_mm
        !! Flag for k-2 point stencil
        integer, intent(out) :: packed_flag
        !! Packed representation of all flags

        if (bflag < 1 .or. bflag > MAX_BLOCKS) then
            call handle_error("Invalid block number", GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Block number:", &
                              [bflag]))
        end if
        if (flag_p < 0 .or. flag_p > 1) then
            call handle_error("Invalid flag for k+1 point stencil", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Flag:", [flag_p]))
        end if
        if (flag_pp < 0 .or. flag_pp > 1) then
            call handle_error("Invalid flag for k+2 point stencil", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Flag:", [flag_pp]))
        end if
        if (flag_m < 0 .or. flag_m > 1) then
            call handle_error("Invalid flag for k-1 point stencil", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Flag:", [flag_m]))
        end if
        if (flag_mm < 0 .or. flag_mm > 1) then
            call handle_error("Invalid flag for k-2 point stencil", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Flag:", [flag_mm]))
        end if

        packed_flag = &
            bflag &
            + flag_p *2**BFLAG_BITS &
            + flag_pp*2**(BFLAG_BITS + FLAG_P_BITS) &
            + flag_m *2**(BFLAG_BITS + FLAG_P_BITS + FLAG_PP_BITS) &
            + flag_mm*2**(BFLAG_BITS + FLAG_P_BITS + FLAG_PP_BITS + FLAG_M_BITS)
    end subroutine

    pure integer function unpack_bflag(packed_flag)
        !! Subroutine to obtain the bflag from the packed representation
        integer, intent(in) :: packed_flag
        !! Packed representation of the flags

        ! Unpack bflag to the range of [1, MAX_BLOCKS]
        unpack_bflag = mod(abs(packed_flag), 2**BFLAG_BITS)
    end function

    pure function unpack_all_flags(packed_flag) result(flags)
        !! Function to unpack all flags: flag_p, flag_pp, flag_m, flag_mm
        !! from the packed representation
        !! The last index contains information on the sign of the packed flag
        integer, intent(in) :: packed_flag
        !! Packed representation of all flags
        logical, dimension(5) :: flags
        !! Array containing all flags

        integer :: abs_val

        abs_val = abs(packed_flag)

        flags(1) = mod(abs_val &
                       / 2**BFLAG_BITS, &
                       2**FLAG_P_BITS) == 0
        flags(2) = mod(abs_val &
                       / 2**(BFLAG_BITS+FLAG_P_BITS), &
                       2**FLAG_PP_BITS) == 0
        flags(3) = mod(abs_val &
                       / 2**(BFLAG_BITS+FLAG_P_BITS+FLAG_PP_BITS), &
                       2**FLAG_M_BITS) == 0
        flags(4) = mod(abs_val &
                       / 2**(BFLAG_BITS+FLAG_P_BITS+FLAG_PP_BITS+FLAG_M_BITS), &
                       2**FLAG_MM_BITS) == 0
        flags(5) = packed_flag < 0
    end function

end module
