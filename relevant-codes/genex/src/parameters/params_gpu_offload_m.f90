module params_gpu_offload_m
    !! Module containing parameters related to GPU offloading
    use, intrinsic :: iso_c_binding, only: C_INT32_T
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use interop_params_m, only: set_cxx_params_gpu_offload
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_GPU
    ! From PARALLAX
    use device_handling_m, only: &
        PARALLAX_BACKEND_CPU   => BACKEND_CPU, &
        PARALLAX_BACKEND_GPU   => BACKEND_GPU, &
        PARALLAX_BACKEND_RLU   => BACKEND_ROCALUTION, &
        PARALLAX_BACKEND_AMGX  => BACKEND_AMGX, &
        PARALLAX_BACKEND_HYPRE => BACKEND_HYPRE

    implicit none
    private

    integer, parameter, public :: GPU_OFFLOAD_CPU = 0
    !! Possible GPU backend parameter: CPU kernel with OpenMP directives
    integer, parameter, public :: GPU_OFFLOAD_ACC = 1
    !! Possible GPU backend parameter: GPU kernel with OpenACC directives
    integer, parameter, public :: GPU_OFFLOAD_OMPX = 2
    !! Possible GPU backend parameter: GPU kernel with OpenMP offload
    integer, parameter, public :: GPU_OFFLOAD_CUDA = 3
    !! Possible GPU backend parameter: GPU kernel with CUDA

    public :: PARALLAX_BACKEND_CPU
    !! Possible PARALLAX GPU backend parameter: CPU kernel
    public :: PARALLAX_BACKEND_GPU
    !! Possible PARALLAX GPU backend parameter: GPU kernel
    public :: PARALLAX_BACKEND_RLU
    !! Possible PARALLAX GPU backend parameter: external kernel via RocALUTION
    public :: PARALLAX_BACKEND_AMGX
    !! Possible PARALLAX GPU backend parameter: external kernel via NVIDIA AMGX
    public :: PARALLAX_BACKEND_HYPRE
    !! Possible PARALLAX GPU backend parameter: external kernel via HYPRE

    logical :: use_gpu_offload = .false.
    !! GPU offloading via Fortran/C++ interface is used only if it is .true.
    integer :: gpu_offload_backend = GPU_OFFLOAD_CPU
    !! Parameter specifying GPU offloading backend

    logical :: use_parallax_gpu_offload = .false.
    !! Fortran/C++ GPU offloading features on PARALLAX are used
    !! only if it is .true.
    integer :: parallax_gpu_offload_backend = PARALLAX_BACKEND_CPU
    !! Parameter specifying GPU offloading backend of PARALLAX
    logical :: use_parallax_gpu_data_explicit = .false.
    !! Explicit CPU/GPU data management on PARALLAX is used only if it is .true.
    logical :: swap_mesh_members = .false.
    !! Swap Fortran/C++ mesh member pointers if it is .true.
    integer :: large_array_alignment = 0
    !! Custom alignment setting for large arrays allocated on C++

    namelist / params_gpu_offload / &
        use_gpu_offload, gpu_offload_backend, use_parallax_gpu_offload, &
        parallax_gpu_offload_backend, use_parallax_gpu_data_explicit, &
        swap_mesh_members, large_array_alignment

    public :: read_params_gpu_offload
    public :: write_params_gpu_offload

    public :: get_use_gpu_offload
    public :: get_gpu_offload_backend
    public :: get_use_parallax_gpu_offload
    public :: get_parallax_gpu_offload_backend
    public :: get_use_parallax_gpu_data_explicit
    public :: get_swap_mesh_members
    public :: get_array_alignment
    public :: check_gpu_functionalities

#ifdef ENABLE_GPU
    interface
        integer(kind=C_INT32_T) function cbind_check_gpu_functionalities() &
            bind(C, name='cbind_check_gpu_functionalities')
            import :: C_INT32_T
        end function
    end interface
#endif

contains

    pure logical function get_use_gpu_offload()
        !! Returns .true. if GPU offloading is used and .false. otherwise
        get_use_gpu_offload = use_gpu_offload
    end function

    pure integer function get_gpu_offload_backend()
        !! Returns the parameter of GPU offloading backend
        get_gpu_offload_backend = gpu_offload_backend
    end function

    pure logical function get_use_parallax_gpu_offload()
        !! Returns .true. if PARALLAX GPU offloading is used
        !! and .false. otherwise
        get_use_parallax_gpu_offload = use_parallax_gpu_offload
    end function

    pure integer function get_parallax_gpu_offload_backend()
        !! Returns the parameter of PARALLAX GPU offloading backend
        get_parallax_gpu_offload_backend = parallax_gpu_offload_backend
    end function

    pure logical function get_use_parallax_gpu_data_explicit()
        !! Returns .true. if PARALLAX explicit CPU/GPU data management is used
        !! and .false. otherwise
        get_use_parallax_gpu_data_explicit = use_parallax_gpu_data_explicit
    end function

    pure logical function get_swap_mesh_members()
        !! Returns .true. if mesh member pointers is swapped between Fortran/C++
        get_swap_mesh_members = swap_mesh_members
    end function

    pure integer function get_array_alignment()
        !! Returns the custom alignment for large array
        get_array_alignment = large_array_alignment
    end function

    subroutine write_params_gpu_offload(filename)
        !! Writes the GPU offload namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_gpu_offload, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="gpu_offload")

        close(iunit)
    end subroutine

    subroutine read_params_gpu_offload(filename)
        !! Reads the GPU oflload namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_gpu_offload, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="gpu_offload", &
                               iunit=iunit)

        close(iunit)

        call check_params_sanity()

        call set_cxx_params_gpu_offload()

    contains

        subroutine check_params_sanity()
            !! Check the sanity of the given GPU offloading related parameters
            if(use_gpu_offload) then
#ifndef ENABLE_GPU
                call handle_error("Cannot use the C++ layer via &
                                  &use_gpu_offload=.true. when the layer is &
                                  &not built: -DENABLE_GPU=OFF", &
                                  GENEX_ERR_GPU, __LINE__, __FILE__)
#endif
            endif

            select case(gpu_offload_backend)
                case(GPU_OFFLOAD_ACC)
#ifndef ENABLE_OPENACC
                    call handle_error("Cannot use the OpenACC backend &
                                      &when it is not built!", &
                                      GENEX_ERR_GPU, __LINE__, __FILE__)
#endif
                case(GPU_OFFLOAD_OMPX)
#ifndef ENABLE_OPENMPX
                    call handle_error("Cannot use the OpenMP offload backend &
                                      &when it is not built!", &
                                      GENEX_ERR_GPU, __LINE__, __FILE__)
#endif
                case(GPU_OFFLOAD_CUDA)
#ifndef ENABLE_CUDA
                    call handle_error("Cannot use the CUDA backend &
                                      &when it is not built!", &
                                      GENEX_ERR_GPU, __LINE__, __FILE__)
#endif
            end select

        end subroutine

    end subroutine

    subroutine check_gpu_functionalities()
        !! Checks if C++ functionallity works as intended on GPUs
        !! with the chosen backend
        integer :: ierr

        if(use_gpu_offload) then
#ifdef ENABLE_GPU
            ierr = cbind_check_gpu_functionalities()
            if(ierr /= 0) then
                call handle_error("Cannot find GPUs or execute kernels &
                                  &on GPUs!", &
                                  GENEX_ERR_GPU, __LINE__, __FILE__)
            endif
#endif
        endif
    end subroutine

end module params_gpu_offload_m
