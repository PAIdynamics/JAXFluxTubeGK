module dcomm_handler_gpu_m
    !! Module containing additional resources related to GPU and Fortran/C++
    !! interoperability for the dcomm_handler_t type
    use, intrinsic :: iso_c_binding, only: C_PTR, C_INT32_T
    implicit none
    private

    public :: cbind_dcomm_handler_initialize
    public :: cbind_dcomm_handler_finalize
    public :: cbind_dcomm_handler_update_device_RZ

    type, public, bind(C) :: dcomm_handler_data_t
        !! Fortran/C++ interoperable structure of the class members
        !! relevant to dcomm_handler_t

        ! Communication part

        integer(C_INT32_T) :: comm_base
        !! Base MPI communicator where to create the topology from
        integer(C_INT32_T) :: comm_cart
        !! MPI communicator for the complete topology
        integer(C_INT32_T) :: comm_phi
        !! MPI communicator for the phi direction
        integer(C_INT32_T) :: comm_vp
        !! MPI communicator for the vp direction
        integer(C_INT32_T) :: comm_mu
        !! MPI communicator for the mu direction
        integer(C_INT32_T) :: comm_sp
        !! MPI communicator for the sp direction
        integer(C_INT32_T) :: comm_phi_sp
        !! MPI communicator for the phi and sp direction
        integer(C_INT32_T) :: comm_vp_mu
        !! MPI communicator for the vp and mu direction
        integer(C_INT32_T) :: comm_phi_vp_mu
        !! MPI communicator for the phi, vp and mu direction
        integer(C_INT32_T) :: comm_vp_mu_sp
        !! MPI communicator for the vp and mu and sp direction
        integer(C_INT32_T) :: comm_mu_sp
        !! MPI communicator for mu and sp direction
        integer(C_INT32_T) :: comm_phi_mu_sp
        !! MPI communicator for phi, mu and sp direction
        integer(C_INT32_T) :: n_procs_total
        !! Number of procs in total
        integer(C_INT32_T) :: n_procs_RZ
        !! Number of procs in RZ direction
        integer(C_INT32_T) :: n_procs_phi
        !! Number of procs in phi direction
        integer(C_INT32_T) :: n_procs_vp
        !! Number of procs in vp direction
        integer(C_INT32_T) :: n_procs_mu
        !! Number of procs in mu direction
        integer(C_INT32_T) :: n_procs_sp
        !! Number of procs in sp direction
        integer(C_INT32_T) :: rank
        !! Rank ID where the code is running on

        ! Domain decomposition part

        integer(C_INT32_T) :: n_dims
        !! Number of MPI parallelized dimensions
        type(C_PTR) :: dim_permut_ptr
        !! C pointer to the array specifying the order of the dimensions
        type(C_PTR) :: number_of_data_elements_ptr
        !! C pointer to the array specifying the number of data elements
        !! for each dimension (excludes ghost cells)
        type(C_PTR) :: number_of_elements_ptr
        !! C pointer to the array specifying the number of elements
        !! for each dimension (includes ghost cells)
        type(C_PTR) :: number_of_ghosts_ptr
        !! C pointer to the array specifying the number of ghost cells
        !! for each dimension
        type(C_PTR) :: lb_ptr
        !! C pointer to the array specifying the lower boundary index
        !! of each dimension
        type(C_PTR) :: ub_ptr
        !! C pointer to the array specifying the upper boundary index
        !! of each dimension
        type(C_PTR) :: lb_stripped_ptr
        !! C pointer to the array specifying the lower boundary index
        !! without ghost of each dimension
        type(C_PTR) :: ub_stripped_ptr
        !! C pointer to the array specifying the upper boundary index
        !! without ghost of each dimension
    end type

    interface
        integer(kind=C_INT32_T) function cbind_dcomm_handler_initialize( &
            dcomm_handler_data, dcomm_handler_cxx_pptr) &
            bind(C, name="cbind_dcomm_handler_initialize")
            !! Fortran/C++ interoperable routine for the initialization of
            !! the dcomm_handler_t C++ class
            import :: C_INT32_T, C_PTR, dcomm_handler_data_t
            type(dcomm_handler_data_t), intent(in) :: dcomm_handler_data
            !! Object of Fortran/C++ interoperable
            !! structure based on dcomm_handler_data_t
            type(C_PTR), intent(inout) :: dcomm_handler_cxx_pptr
            !! C pointer to the dcomm_handler_t C++ class instance pointer
        end function

        integer(kind=C_INT32_T) function cbind_dcomm_handler_finalize( &
            dcomm_handler_cxx_pptr) &
            bind(C, name="cbind_dcomm_handler_finalize")
            !! Fortran/C++ interoperable routine for the finalization of
            !! dcomm_handler_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: dcomm_handler_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_dcomm_handler_update_device_RZ( &
            dcomm_handler_cxx_pptr) &
            bind(C, name="cbind_dcomm_handler_update_device_RZ")
            !! Fortran/C++ interoperable routine for updating the RZ domain
            !! on GPU from CPU
            import ::  C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: dcomm_handler_cxx_pptr
        end function
    end interface

end module
