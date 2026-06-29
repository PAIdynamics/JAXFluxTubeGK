module genex_error_handling_m
    !! Module to handle errors in a standardized way

    ! NOTE: When using handle_error with preprocessor macros __LINE__ and
    !       __FILE__, it is important that any error message string that has
    !       been continued on a new line with &, is NOT in the same line as
    !       these macros. This is due to issues with the GNU compiler.
    !
    !       A VALID use example is:
    !       call handle_error("Error &
    !                         &message", ERR_CODE, &
    !                         __LINE__, __FILE__)
    !
    !       An INVALID use example is:
    !       call handle_error("Error &
    !                         &message", ERR_CODE, __LINE__, __FILE__)

    use genex_status_codes_m, only: GENEX_ERR_NETCDF, &
                                    GENEX_ERR_GPU, &
                                    GENEX_ERR_VSPEC, &
                                    GENEX_ERR_NOT_IMPLEMENTED

    ! From PARALLAX
    use error_handling_m, only: error_handler_t, error_info_t, ERRORS_ARE_FATAL

    implicit none

    private

    type(error_handler_t) :: genex_error_handler = &
        error_handler_t(handler_name="GENEX", &
                        error_mode=ERRORS_ARE_FATAL, &
                        netcdf_error_code=GENEX_ERR_NETCDF)
    !! Stores the setup information about GENEX errors. Is used by the
    !! the public subroutines in this module to deliver the error handling
    !! functionality to GENEX.

    public :: error_info_t

    public :: set_error_mode
    public :: handle_error
    public :: handle_error_netcdf
    public :: handle_error_gpu
    public :: handle_error_vspec
    public :: handle_error_notimp

    public :: GPU_ERR_NONE, GPU_ERR_INITIALIZE, &
              GPU_ERR_FINALIZE, GPU_ERR_COPY, GPU_ERR_START_EXCH, &
              GPU_ERR_FINISH_EXCH
    enum, bind(C) ! GPU_ERR
        !! Enumerator defining the GPU error codes which are used to select
        !! standardized error messages
        enumerator :: GPU_ERR_NONE        = 0
        enumerator :: GPU_ERR_INITIALIZE  = 1
        enumerator :: GPU_ERR_FINALIZE    = 2
        enumerator :: GPU_ERR_COPY        = 3
        enumerator :: GPU_ERR_START_EXCH  = 4
        enumerator :: GPU_ERR_FINISH_EXCH = 5
    end enum

    public :: VSPEC_ERR_NONE, VSPEC_ERR_INITIALIZE
    enum, bind(C) ! VSPEC_ERR
        enumerator :: VSPEC_ERR_NONE        = 0
        enumerator :: VSPEC_ERR_INITIALIZE  = 1
    end enum

contains

    subroutine set_error_mode(err_mode)
        !! Sets the way GENEX responds to internal errors.
        !! When set to errors return, the error is logged into stderr without
        !! program termination, expecting that the error will be handled
        !! by the calling routine. When set to errors are fatal, the code
        !! will be terminated.
        integer, intent(in) :: err_mode
        !! Error mode: Use ERRORS_RETURN or ERRORS_ARE_FATAL from PARALLAX.
        !!             Other values are ignored and ERRORS_ARE_FATAL is used.

        call genex_error_handler%set_error_mode(err_mode)
    end subroutine

    subroutine handle_error(message, status_code, line_number, file_name, &
                            additional_info)
        !! Logs an error and stops the program. If status_code is
        !! GENEX_SUCCESS, this subroutine will do nothing.
        !!
        !! If status_code has an undefined value, the subroutine will be
        !! executed but the undefined status_code will be printed.
        !! This usage should be avoided, instead valid status codes given in
        !! status_codes_m should be used.
        character(len=*), intent(in) :: message
        !! Error or warning message
        integer, intent(in) :: status_code
        !! Error or warning code
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__
        type(error_info_t), optional, intent(in) :: additional_info
        !! Additional information

        ! NOTE: We assume that GENEX_SUCCESS=PARALLAX_SUCCESS by using the
        !       error handler. Further, the stderr output of PARALLAX has to
        !       be redirected to the same err channel as used by GENEX.
        call genex_error_handler%handle_error(message, status_code, &
                                              line_number, file_name, &
                                              additional_info)
    end subroutine

    subroutine handle_error_netcdf(istatus, line_number, file_name)
        !! Wrapper of handle error for calls of NetCDF functions.
        !! Checks for NetCDF errors and prints a standardized error message.
        integer, intent(in) :: istatus
        !! Status integer returned by the NetCDF function
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        call genex_error_handler%handle_error_netcdf(istatus, line_number, &
                                                     file_name)
    end subroutine

    subroutine handle_error_gpu(istatus, line_number, file_name)
        !! Wrapper of handle error for calls of GPU cbind functions.
        !! Checks for GPU errors and prints a standardized error message.
        integer, intent(in) :: istatus
        !! Status integer which should have values given by the GPU_ERR enum
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        character(len=:), allocatable :: err_message

        select case(istatus)
            case(GPU_ERR_INITIALIZE)
                err_message = "Failed to initialize GPU instance. Either the &
                              &C++ class is not on the device or an &
                              &undefined GPU backend is used!"
            case(GPU_ERR_FINALIZE)
                err_message = "Failed to finalize GPU instance. Either the &
                              &C++ class is still on the device or an &
                              &undefined GPU backend is used!"
            case(GPU_ERR_COPY)
                err_message = "Failed to copy data to GPU. Either an &
                              &array is not copied to the GPU or an &
                              &undefined GPU backend is used!"
            case(GPU_ERR_START_EXCH)
                err_message = "Failed to start exchange on GPU!"
            case(GPU_ERR_FINISH_EXCH)
                err_message = "Failed to finish exchange on GPU!"
            case default
                return
        end select

        call genex_error_handler%handle_error(err_message, GENEX_ERR_GPU, &
                                              line_number, file_name)
    end subroutine

    subroutine handle_error_vspec(istatus, line_number, file_name)
        !! Wrapper of handle error for calls of vspec operator.
        !! Checks for vspec errors and prints a standardized error message.
        integer, intent(in) :: istatus
        !! Status integer which should have values given by the VSPEC_ERR enum
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        character(len=:), allocatable :: err_message

        select case(istatus)
            case(VSPEC_ERR_INITIALIZE)
                err_message = "Failed to initialize vspec operator. &
                              &It requires get_use_vspectral = .true. &
                              &to be used!"
            case default
                return
        end select

        call genex_error_handler%handle_error(err_message, GENEX_ERR_VSPEC, &
                                              line_number, file_name)

    end subroutine

    subroutine handle_error_notimp(feature, line_number, file_name)
        !! Wrapper to throw errors that occur if not implemented features
        !! were tried to be accessed.
        character(len=*), intent(in) :: feature
        !! Name of the feature that was tried to be accessed
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        character(len=:), allocatable :: err_message

        err_message = "Tried to use feature '"//feature//"' which is not &
                      &implemented in this version of the code!"

        call genex_error_handler%handle_error(err_message, &
                                              GENEX_ERR_NOT_IMPLEMENTED, &
                                              line_number, file_name)
    end subroutine

end module
