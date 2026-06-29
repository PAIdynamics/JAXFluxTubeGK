submodule (dcomm_handler_m) dcomm_handler_s
    use mpi
    use, intrinsic :: iso_c_binding, only: c_loc
    use logger_m, only: logger_get_debug_channel
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_COMMUNICATION
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE, GPU_ERR_COPY
    use dimensions_m, only: DIM_RZ, DIM_PHI, DIM_VP, DIM_MU, DIM_SP
    use params_gpu_offload_m, only : get_use_gpu_offload
    use params_mesh_m, only: get_n_points_phi, get_n_points_vp, &
                             get_n_points_mu, get_n_points_sp, &
                             get_use_vspectral

    use params_collisions_m, only: get_coll_type
#ifdef ENABLE_GPU
    use dcomm_handler_gpu_m, only: dcomm_handler_data_t, &
                                   cbind_dcomm_handler_initialize, &
                                   cbind_dcomm_handler_finalize, &
                                   cbind_dcomm_handler_update_device_RZ
#endif

    ! From PARALLAX
    use screen_io_m, only: set_parallax_stdout => set_stdout

    implicit none

contains

    module subroutine initialize_dcomm_handler(this, comm, &
                                               n_procs_phi, n_procs_vp, &
                                               n_procs_mu, n_procs_sp, &
                                               dim_permut)
        class(dcomm_handler_t), intent(inout) :: this
        integer, intent(in) :: comm
        integer, intent(in) :: n_procs_phi
        integer, intent(in) :: n_procs_mu
        integer, intent(in) :: n_procs_vp
        integer, intent(in) :: n_procs_sp
        integer, dimension(:), optional, intent(in) :: dim_permut

        call this%initialize_communication(comm, n_procs_phi, n_procs_vp, &
                                           n_procs_mu, n_procs_sp)

        if (present(dim_permut)) then
            call this%initialize_pvm_domain(dim_permut)
        else
            call this%initialize_pvm_domain([1, 2, 3, 4, 5])
        endif

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Initialize the dcomm_handler_t C++ class
            call this%initialize_gpu(comm)
        endif
#endif
        this%is_initialized_comm = .true.
        ! Only partial initialization up to this point. Need to call
        ! initialize_RZ_domain as well for full init of the domain
        ! decomposition.
        this%is_initialized_domain = .false.
    end subroutine

    module subroutine initialize_communication(this, comm, &
                                               n_procs_phi, n_procs_vp, &
                                               n_procs_mu, n_procs_sp)
        class(dcomm_handler_t), intent(inout) :: this
        integer, intent(in) :: comm
        integer, intent(in) :: n_procs_phi
        integer, intent(in) :: n_procs_mu
        integer, intent(in) :: n_procs_vp
        integer, intent(in) :: n_procs_sp

        ! NOTE: Currently the parallelization in the RZ dimension is not
        !       implemented. Thus we set the number of processes to 1.
        integer, parameter :: n_procs_RZ = 1
        logical, parameter :: use_reorder = .false.

        integer :: ierr
        logical :: periods(n_dims) = .false.
        logical :: activated_dims(n_dims)
        integer :: procs_per_dim(n_dims)

        call this%check_parallelization(comm, n_procs_RZ, n_procs_phi, &
                                        n_procs_vp, n_procs_mu, n_procs_sp)

        periods(DIM_PHI) = .true.
        procs_per_dim(DIM_RZ)  = n_procs_RZ
        procs_per_dim(DIM_PHI) = n_procs_phi
        procs_per_dim(DIM_MU)  = n_procs_mu
        procs_per_dim(DIM_VP)  = n_procs_vp
        procs_per_dim(DIM_SP)  = n_procs_sp

        call mpi_cart_create(comm, n_dims, procs_per_dim, periods, &
                             use_reorder, this%comm_cart, ierr)
        call mpi_cart_coords(this%comm_cart, this%rank, n_dims, &
                             this%cart_coords, ierr)

        ! Create sub communicators
        activated_dims = .false.
        activated_dims(DIM_PHI) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_phi, ierr)

        activated_dims = .false.
        activated_dims(DIM_VP) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_vp, ierr)

        activated_dims = .false.
        activated_dims(DIM_MU) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_mu, ierr)

        activated_dims = .false.
        activated_dims(DIM_SP) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_sp, ierr)

        activated_dims = .false.
        activated_dims(DIM_PHI) = .true.
        activated_dims(DIM_SP)  = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_phi_sp, &
                          ierr)

        activated_dims = .false.
        activated_dims(DIM_VP) = .true.
        activated_dims(DIM_MU) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_vp_mu, ierr)

        activated_dims = .false.
        activated_dims(DIM_PHI) = .true.
        activated_dims(DIM_VP)  = .true.
        activated_dims(DIM_MU)  = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_phi_vp_mu, &
                          ierr)

        activated_dims = .false.
        activated_dims(DIM_VP) = .true.
        activated_dims(DIM_MU) = .true.
        activated_dims(DIM_SP) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_vp_mu_sp, &
                          ierr)

        activated_dims = .false.
        activated_dims(DIM_MU) = .true.
        activated_dims(DIM_SP) = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_mu_sp, &
                          ierr)

        activated_dims = .false.
        activated_dims(DIM_PHI) = .true.
        activated_dims(DIM_MU)  = .true.
        activated_dims(DIM_SP)  = .true.
        call mpi_cart_sub(this%comm_cart, activated_dims, this%comm_phi_mu_sp, &
                          ierr)
    end subroutine

#ifdef ENABLE_GPU
    module subroutine initialize_gpu(this, comm)
        class(dcomm_handler_t), intent(inout) :: this
        integer, intent(in) :: comm

        type(dcomm_handler_data_t), allocatable :: dcomm_handler_data
        ! Fortran/C++ interoperable structure for dcomm_handler_t class members
        integer :: ierr
        ! C++ error status

        ! Expose class members to the coressponding interoperable structure
        allocate(dcomm_handler_data)
        call expose_data(this, comm, dcomm_handler_data)

        ! Initialize dcomm_handler_t C++ class, including deep copy to
        ! the device
        ierr = cbind_dcomm_handler_initialize(dcomm_handler_data, &
                                              this%dcomm_handler_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif

        deallocate(dcomm_handler_data)

    contains

        subroutine expose_data(this, comm, dcomm_handler_data)
            !! Expose class members of dcomm_handler_t to a Fortran/C++
            !! interoperable structure based on dcomm_handler_data_t
            class(dcomm_handler_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: comm
            !! Base MPI communicator where to create the topology from
            type(dcomm_handler_data_t), intent(inout) :: dcomm_handler_data
            !! Fortran/C++ interoperable structure of dcomm_handler_t

            ! Communication part

            dcomm_handler_data%comm_base      = comm
            dcomm_handler_data%comm_cart      = this%comm_cart
            dcomm_handler_data%comm_phi       = this%comm_phi
            dcomm_handler_data%comm_vp        = this%comm_vp
            dcomm_handler_data%comm_mu        = this%comm_mu
            dcomm_handler_data%comm_sp        = this%comm_sp
            dcomm_handler_data%comm_phi_sp    = this%comm_phi_sp
            dcomm_handler_data%comm_vp_mu     = this%comm_vp_mu
            dcomm_handler_data%comm_phi_vp_mu = this%comm_phi_vp_mu
            dcomm_handler_data%comm_vp_mu_sp  = this%comm_vp_mu_sp
            dcomm_handler_data%comm_mu_sp     = this%comm_mu_sp
            dcomm_handler_data%comm_phi_mu_sp = this%comm_phi_mu_sp
            dcomm_handler_data%n_procs_total  = this%n_procs_total
            dcomm_handler_data%n_procs_RZ     = this%n_procs_RZ
            dcomm_handler_data%n_procs_phi    = this%n_procs_phi
            dcomm_handler_data%n_procs_vp     = this%n_procs_vp
            dcomm_handler_data%n_procs_mu     = this%n_procs_mu
            dcomm_handler_data%n_procs_sp     = this%n_procs_sp
            dcomm_handler_data%rank           = this%rank

            ! Domain decomposition part

            dcomm_handler_data%n_dims = n_dims
            dcomm_handler_data%dim_permut_ptr = c_loc(this%dim_permut)
            dcomm_handler_data%number_of_data_elements_ptr = &
                c_loc(this%number_of_data_elements)
            dcomm_handler_data%number_of_elements_ptr = &
                c_loc(this%number_of_elements)
            dcomm_handler_data%number_of_ghosts_ptr = &
                c_loc(this%number_of_ghosts)
            dcomm_handler_data%lb_ptr = c_loc(this%lb)
            dcomm_handler_data%ub_ptr = c_loc(this%ub)
            dcomm_handler_data%lb_stripped_ptr = c_loc(this%lb_stripped)
            dcomm_handler_data%ub_stripped_ptr = c_loc(this%ub_stripped)
        end subroutine

    end subroutine

    module subroutine finalize_gpu(this)
        class(dcomm_handler_t), intent(inout) :: this

        integer :: ierr
        ! C++ error status

        ierr = cbind_dcomm_handler_finalize(this%dcomm_handler_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine
#endif

    module subroutine check_parallelization(this, comm, n_procs_RZ, &
                                            n_procs_phi, n_procs_vp, &
                                            n_procs_mu, n_procs_sp)
        class(dcomm_handler_t), intent(inout) :: this
        integer, intent(in) :: comm
        integer, intent(in) :: n_procs_RZ
        integer, intent(in) :: n_procs_phi
        integer, intent(in) :: n_procs_mu
        integer, intent(in) :: n_procs_vp
        integer, intent(in) :: n_procs_sp

        integer :: n_procs_found, ierr

        if(n_procs_RZ /= 1) then
            call handle_error("In-plane (RZ) MPI decomposition is not &
                              &supported at the moment!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__)
        endif

        if(n_procs_phi < 1 .or. n_procs_vp < 1 &
           .or. n_procs_mu < 1 .or. n_procs_sp < 1) then
            call handle_error("A given number of MPI procs is < 1!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Was (phi, vp, mu, sp): ", &
                                                [n_procs_phi, n_procs_vp, &
                                                 n_procs_mu, n_procs_sp]))
        endif

        this%n_procs_RZ    = n_procs_RZ
        this%n_procs_phi   = n_procs_phi
        this%n_procs_mu    = n_procs_mu
        this%n_procs_vp    = n_procs_vp
        this%n_procs_sp    = n_procs_sp
        this%n_procs_total = n_procs_RZ * n_procs_phi * n_procs_vp &
                                        * n_procs_mu  * n_procs_sp

        call mpi_comm_size(comm, n_procs_found, ierr)
        call mpi_comm_rank(comm, this%rank, ierr)

        if (this%n_procs_total /= n_procs_found) then
            call handle_error("Number of MPI procs does not match &
                              &chosen parallelization!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Total/found: ", &
                                                [this%n_procs_total, &
                                                 n_procs_found]))
        endif
    end subroutine

    module subroutine initialize_pvm_domain(this, dim_permut)
        class(dcomm_handler_t), intent(inout) :: this
        integer, dimension(:), intent(in) :: dim_permut

        integer :: ind
        integer, dimension(n_dims) :: n_procs, coords, n_points, &
                                      n_points_local, n_ghost

        ! Determine the total domain bounds
        n_points(DIM_RZ)  = 0
        n_points(DIM_PHI) = get_n_points_phi()
        n_points(DIM_VP)  = get_n_points_vp()
        n_points(DIM_MU)  = get_n_points_mu()
        n_points(DIM_SP)  = get_n_points_sp()

        ! Determine the bounds of the MPI domain decomposition
        n_procs(DIM_RZ)  = this%get_n_procs_RZ()
        n_procs(DIM_PHI) = this%get_n_procs_phi()
        n_procs(DIM_VP)  = this%get_n_procs_vp()
        n_procs(DIM_MU)  = this%get_n_procs_mu()
        n_procs(DIM_SP)  = this%get_n_procs_sp()

        ! Determine the number of ghosts
        n_ghost(DIM_RZ)  = 0
        n_ghost(DIM_PHI) = 2
        n_ghost(DIM_VP)  = 2
        n_ghost(DIM_SP)  = 0
        if(get_coll_type() == "lbd" &
           .or. get_coll_type() == "lorentz" &
           .or. get_coll_type() == "fpl" &
           .or. get_use_vspectral()) then
            n_ghost(DIM_MU) = 1
        else
            n_ghost(DIM_MU) = 0
        endif

        ! Check if the chosen parallelization complies with the number of
        ! points
        if(modulo(n_points(DIM_PHI), n_procs(DIM_PHI)) /= 0) then
            call handle_error("Number of phi points is not divisible by &
                              &number of phi procs!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Points/procs: ", &
                                                [n_points(DIM_PHI), &
                                                 n_procs(DIM_PHI)]))
        endif
        if(modulo(n_points(DIM_VP), n_procs(DIM_VP)) /= 0) then
            call handle_error("Number of vp points is not divisible by &
                              &number of vp procs!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Points/procs: ", &
                                                [n_points(DIM_VP), &
                                                 n_procs(DIM_VP)]))
        endif
        if(modulo(n_points(DIM_MU), n_procs(DIM_MU)) /= 0) then
            call handle_error("Number of mu points is not divisible by &
                              &number of mu procs!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Points/procs: ", &
                                                [n_points(DIM_MU), &
                                                 n_procs(DIM_MU)]))
        endif
        if(modulo(n_points(DIM_SP), n_procs(DIM_SP)) /= 0) then
            call handle_error("Number of sp points is not divisible by &
                              &number of sp procs!", &
                              GENEX_ERR_COMMUNICATION, __LINE__, __FILE__, &
                              additional_info=error_info_t(&
                                                "Points/procs: ", &
                                                [n_points(DIM_SP), &
                                                 n_procs(DIM_SP)]))
        endif

        do ind = 1, n_dims
            n_points_local(ind) = n_points(ind) / n_procs(ind)
        enddo
        coords = this%get_cart_coords()

        ! Set domain decomposed bounds, we index the bounds from 1 on
        do ind = 1, n_dims
            this%dim_permut(ind) = dim_permut(ind)
            if(ind == DIM_RZ) cycle

            this%lb_stripped(dim_permut(ind)) &
                = n_points_local(ind) * coords(ind) + 1
            this%ub_stripped(dim_permut(ind)) &
                = n_points_local(ind) * (coords(ind) + 1)

            this%number_of_ghosts(dim_permut(ind)) = n_ghost(ind)

            this%lb(dim_permut(ind)) &
                = this%lb_stripped(dim_permut(ind)) - n_ghost(ind)

            this%ub(dim_permut(ind)) &
                = this%ub_stripped(dim_permut(ind)) + n_ghost(ind)

            this%number_of_data_elements(dim_permut(ind)) &
                = this%ub_stripped(dim_permut(ind)) &
                - this%lb_stripped(dim_permut(ind)) + 1

            this%number_of_elements(dim_permut(ind)) &
                = this%ub(dim_permut(ind)) - this%lb(dim_permut(ind)) + 1
        enddo
    end subroutine

    module subroutine initialize_RZ_domain(this, n_points_RZ)
        class(dcomm_handler_t), intent(inout) :: this
        integer, intent(in) :: n_points_RZ

        integer :: n_procs, n_points_local, coords(5), ierr

        n_procs = this%get_n_procs_RZ()
        n_points_local = n_points_RZ / n_procs
        coords = this%get_cart_coords()

        this%lb_stripped(this%dim_permut(DIM_RZ)) &
                                        = n_points_local * coords(DIM_RZ) + 1
        this%ub_stripped(this%dim_permut(DIM_RZ)) &
                                        = n_points_local * (coords(DIM_RZ) + 1)

        ! The RZ dimension has no MPI ghosts
        this%number_of_ghosts(this%dim_permut(DIM_RZ)) = 0

        ! The RZ dimension has no negative ghost cells because it is
        ! stored as an unstructured grid
        this%lb(this%dim_permut(DIM_RZ)) &
                                    = this%lb_stripped(this%dim_permut(DIM_RZ))
        this%ub(this%dim_permut(DIM_RZ)) &
                                    = this%ub_stripped(this%dim_permut(DIM_RZ))

        this%number_of_data_elements(this%dim_permut(DIM_RZ)) &
            = this%ub_stripped(this%dim_permut(DIM_RZ)) &
            - this%lb_stripped(this%dim_permut(DIM_RZ)) + 1

        this%number_of_elements(this%dim_permut(DIM_RZ)) &
                                    = this%ub(this%dim_permut(DIM_RZ)) &
                                    - this%lb(this%dim_permut(DIM_RZ)) + 1

#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ! Update the RZ domain on GPU from CPU
            ierr = cbind_dcomm_handler_update_device_RZ( &
                   this%dcomm_handler_cxx_pptr)

            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
        endif
#endif
        this%is_initialized_domain = .true.

        ! Store global problem size
        this%n_points_global_RZ = n_points_RZ
        this%n_points_global_phi = get_n_points_phi()
        this%n_points_global_vp = get_n_points_vp()
        this%n_points_global_mu = get_n_points_mu()
        this%n_points_global_sp = get_n_points_sp()
    end subroutine

    module subroutine finalize(this)
        type(dcomm_handler_t), intent(inout) :: this
        integer :: ierr

        if(this%is_initialized_comm) then

#ifdef ENABLE_GPU
            if(get_use_gpu_offload()) then
                ! Finalize the dcomm_handler_t C++ class
                call this%finalize_gpu()
            endif
#endif
            call mpi_comm_free(this%comm_phi_mu_sp, ierr)
            call mpi_comm_free(this%comm_mu_sp, ierr)
            call mpi_comm_free(this%comm_vp_mu_sp, ierr)
            call mpi_comm_free(this%comm_phi_vp_mu, ierr)
            call mpi_comm_free(this%comm_vp_mu, ierr)
            call mpi_comm_free(this%comm_phi_sp, ierr)
            call mpi_comm_free(this%comm_sp, ierr)
            call mpi_comm_free(this%comm_vp, ierr)
            call mpi_comm_free(this%comm_mu, ierr)
            call mpi_comm_free(this%comm_phi, ierr)
            call mpi_comm_free(this%comm_cart, ierr)
        endif
    end subroutine

    module subroutine print_mesh_info(this)
        class(dcomm_handler_t), intent(inout) :: this
        integer :: debug_channel, ierr
        integer :: ind
        character(len=80) :: fmt_tmp

        debug_channel = logger_get_debug_channel()
        if(this%is_master()) then
            ! Print global problem size info
            write(debug_channel, *)
            write(debug_channel, *) "Global problem size: "
            write(debug_channel, "(A12, 5(' | ', A12))") &
                    "dimension   ", "DIM_RZ", "DIM_PHI", &
                    "DIM_VP", "DIM_MU", "DIM_SP"
            write(debug_channel, "(12X, 5(' | ', I12))") &
                    this%n_points_global_RZ, &
                    this%n_points_global_phi, this%n_points_global_vp, &
                    this%n_points_global_mu, this%n_points_global_sp
            write(debug_channel, "(87('-'))")
            write(debug_channel, "(A36, I20)") "Global number of points &
                    &(total):    ", this%get_size() * &
                    int(this%n_procs_total, kind=INT64)
            write(debug_channel, "(A36, I20)") "Global number of points &
                    &(non-ghost):", &
                    this%get_size_stripped() * &
                    int(this%n_procs_total, kind=INT64)
            write(debug_channel, "(A36, I20)") "Global number of points &
                    &(ghost):    ", &
                    (this%get_size() - this%get_size_stripped()) * &
                    int(this%n_procs_total, kind=INT64)
            write(debug_channel, *)

            ! Print local problem size info
            write(debug_channel, *) "Local number of points for each dimension:"
            write(debug_channel, "(A12, 5(' | ', I12))") "total       ", &
                        this%number_of_elements(1:n_dims)
            write(debug_channel, "(A12, 5(' | ', I12))") "non-ghost   ", &
                        this%number_of_data_elements(1:n_dims)
            write(debug_channel, "(A12, 5(' | ', I12))") "ghost       ", &
                        this%number_of_elements(1:n_dims) - &
                        this%number_of_data_elements(1:n_dims)

            write(debug_channel, "(87('-'))")
            write(debug_channel, "(A36, I20)") "Local number of points &
                    &(total):    ", &
                    this%get_size()
            write(debug_channel, "(A36, I20)") "Local number of points &
                    &(non-ghost):", &
                    this%get_size_stripped()
            write(debug_channel, "(A36, I20)") "Local number of points &
                    &(ghost):    ", &
                    this%get_size() - this%get_size_stripped()
            write(debug_channel, *)
        endif

        call mpi_barrier(this%comm_cart, ierr)

    end subroutine

end submodule
