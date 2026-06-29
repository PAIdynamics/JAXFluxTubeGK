submodule (profiler_m) profiler_s
    use mpi
    use params_gpu_offload_m, only: get_use_gpu_offload
    use params_devtools_m, only: get_debug_profregion, get_isolate_tsync
    use logger_m, only: logger_log_region
    implicit none
contains

    module subroutine profiler_reset()
        call profiler_region_tree%reset()
        is_initialized = .false.
    end subroutine profiler_reset

    module subroutine profiler_start(region_name, ierr, set_path)
        character(len=*), intent(in) :: region_name
        integer, intent(out) :: ierr
        logical, optional, intent(in) :: set_path

        type(profiler_region_t), pointer :: region
        !! Pointer to region which will be started
        logical :: set_path_local
        !! Local variable to give optional set_path a default value

        ierr = GENEX_SUCCESS

        ! Check for illegal characters in region name
        if(index(region_name, ".") /= 0) then
            call handle_error("Profiler region name """//region_name &
                              //""" contains illegal character "".""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
            goto 100
        endif
        if(index(region_name, "/") /= 0) then
            call handle_error("Profiler region name """//region_name &
                              //""" contains illegal character ""/""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
            goto 100
        endif

        ! At first time call, we have to initialize the root of the tree.
        if(.not. is_initialized) then
            call profiler_region_tree%initialize(region_name)
            current_node => profiler_region_tree
            current_path = profiler_region_tree%region_name
            is_initialized = .true.
        endif

        ! Log the current region for debug
        if(get_debug_profregion()) then
            call logger_log_region(trim(current_path)//"/"//trim(region_name), &
                                        ierr)
        endif

        ! Find the performance region
        call find_region(region_name, current_node, region, ierr)
        ! Create region if not exists
        if (ierr /= GENEX_SUCCESS) then
            call create_region(region_name, region)
            ierr = GENEX_SUCCESS
        endif

        ! Start the profiler region
        call region%start(ierr)
        if (ierr /= GENEX_SUCCESS) goto 100

100     if (ierr /= GENEX_SUCCESS) then
            if (error_mode == PROFILER_ERR_RETURN) ierr = GENEX_WRN_PROFILER
            call handle_error("Profiler start failed!", ierr, __LINE__, &
                              __FILE__)
            return
        endif

        ! Set current path to started region if present
        set_path_local = .false.
        if (present(set_path)) set_path_local = set_path
        if (set_path_local) call profiler_set_path("./"//region_name, ierr)

    end subroutine profiler_start

    module subroutine profiler_stop(region_name, ierr, set_path)
        character(len=*), intent(in) :: region_name
        integer, intent(out) :: ierr
        logical, optional, intent(in) :: set_path

        type(profiler_region_t), pointer :: region
        !! Pointer to region which will be stopped
        logical :: set_path_local
        !! Local variable to give optional set_path a default value

        ierr = GENEX_SUCCESS

        ! Set current path to the region above if present
        set_path_local = .false.
        if (present(set_path)) set_path_local = set_path
        if (set_path_local) call profiler_set_path("../", ierr)

        ! Find the performance region given by name
        call find_region(region_name, current_node, region, ierr)

        ! Error if not found
        if (ierr == GENEX_SUCCESS) then
            ! Stop the profiler region
            call region%stop(ierr)
        else
            call handle_error("Profiler region """//region_name &
                              //""" not found in current path """ &
                              //trim(current_node%region_name)//"""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
        endif

        if (ierr /= GENEX_SUCCESS) then
            if (error_mode == PROFILER_ERR_RETURN) ierr = GENEX_WRN_PROFILER
            call handle_error("Profiler stop failed!", ierr, __LINE__, &
                              __FILE__)
            return
        endif
    end subroutine profiler_stop

    module subroutine profiler_print(dcomm_handler, channel, ref_region_name, &
                                     ierr, discard_first_call)
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        integer, intent(in) :: channel
        character(len=*), intent(in) :: ref_region_name
        integer, intent(out) :: ierr
        logical, optional, intent(in) :: discard_first_call

        integer :: ref_calls
        !! Total number of profiler calls from all mpi processes
        real(kind = DP) :: ref_time
        !! Total time of profiler calls from all mpi processes
        logical :: discard_first
        !! If true, the first call will be discarded from the measurement
        type(profiler_region_t), pointer :: ref_region
        !! Pointer to reference region

        ierr = GENEX_SUCCESS

        ! Check for something to print
        if(profiler_region_tree%number_of_children == 0) return

        ! If discard_first_call optional present
        if(present(discard_first_call)) then
            discard_first = discard_first_call
        else
            discard_first = .false.
        endif

        ! Find the reference region
        call find_region(ref_region_name, current_node, ref_region, ierr)
        if (ierr /= GENEX_SUCCESS) then
            call handle_error("Profiler reference region """//ref_region_name &
                              //""" not found in current path """ &
                              //trim(current_node%region_name)//"""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            goto 200
        endif

        ! Get reference values
        call ref_region%exchange(dcomm_handler, discard_first, &
                                 ref_calls, ref_time, ierr)
        if (ierr /= GENEX_SUCCESS) then
            call handle_error("Profiler exchange failed in reference &
                              &region!", GENEX_WRN_PROFILER, &
                              __LINE__, __FILE__)
            goto 200
        endif

        ! Print header of the performance summary
        if(dcomm_handler%is_master()) then
            ! We explicitly declare the first character to be of length clen
            ! to align it to the left of the output.
            write(channel, "(X, A37,' | ',A8,' | ',A14,' | ',A14,' | ',A14)") &
                  [character(len=clen) :: "region"], &
                  "#calls", "time / s", "time per call", "rel.time / %"
            write(channel, "(X, 99('-'))") ! Horizontal line
        endif

        ! Call recursive print of the profiler tree
        call profiler_region_tree%print(dcomm_handler, channel, ref_time, &
                                        lvl=0, discard_first=discard_first, &
                                        ierr=ierr)
        if (ierr /= GENEX_SUCCESS) goto 200

200     if (ierr /= GENEX_SUCCESS) then
            if (error_mode == PROFILER_ERR_RETURN) ierr = GENEX_WRN_PROFILER
            call handle_error("Profiler print failed!", ierr, __LINE__, &
                              __FILE__)
            return
        endif

        if(dcomm_handler%is_master()) then
            write(channel, "(100('-'))") ! Horizontal line
        endif
    end subroutine profiler_print

    module subroutine profiler_set_path(path_name, ierr)
        character(len=*), intent(in) :: path_name
        integer, intent(out) :: ierr

        integer :: idx
        !! Index
        type(profiler_region_t), pointer :: node
        !! Node that will become the current node
        character(len=clen_big) :: first, last
        !! Strings used when splitting the path name in the relative mode
        character(len=clen_big) :: local_path_name
        !! Path name placeholder with appropriate length

        ierr = GENEX_SUCCESS

        ! Copy path name to local placeholder. Below we check the first few
        ! elements of the path name for special characters. Copy to a character
        ! of fixed length ensure that these character elements are indeed
        ! present in the check.
        local_path_name = path_name

        ! If not initialized, try to initialize the tree with the path name.
        if(.not. is_initialized) then
            ! Only do this if no path separator is present. Otherwise there
            ! has been incorrect use of the profiler and we will not initialize
            ! which will result in error below.
            idx = index(local_path_name, "/")
            if (idx == 0) then
                call profiler_region_tree%initialize(local_path_name)
                current_node => profiler_region_tree
                current_path = profiler_region_tree%region_name
                is_initialized = .true.
            else
                ierr = GENEX_ERR_PROFILER
                goto 300
            endif
        endif

        ! Find the node

        ! Relative mode - go one level deeper
        if(local_path_name(1:2) == "./") then
            ! Assumes the region name is given after the "./"
            call find_region(local_path_name(3:), current_node, node, ierr)
            if (ierr /= GENEX_SUCCESS) goto 400
            current_path = trim(current_path)//"/"//trim(node%region_name)

        ! Relative mode - go one level back
        elseif(local_path_name(1:3) == "../") then
            ! Check that name is indeed valid (rest needs to be empty)
            if(trim(local_path_name(4:)) /= "") then
                ierr = GENEX_ERR_PROFILER
                goto 400
            endif
            ! Split of the last level of the current path. Then find this
            ! absolute path
            call split_path_name(current_path, first, last, back=.true.)
            call find_node(first, node, ierr)
            if (ierr /= GENEX_SUCCESS) goto 400
            current_path = first

        ! Absolute mode
        else
            call find_node(local_path_name, node, ierr)
            if (ierr /= GENEX_SUCCESS) goto 400
            current_path = local_path_name
        endif

        ! Set current node pointer
        current_node => node

300     if (ierr /= GENEX_SUCCESS) then
            call handle_error("Invalid name """//trim(local_path_name) &
                              //""" for profiler root path chosen!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            goto 500
        endif

400     if (ierr /= GENEX_SUCCESS) then
            call handle_error("Profiler path """//trim(local_path_name) &
                              //""" does not exist!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            goto 500
        endif

500     if (ierr /= GENEX_SUCCESS) then
            if (error_mode == PROFILER_ERR_RETURN) ierr = GENEX_WRN_PROFILER
            call handle_error("Profiler set_path failed!", ierr, __LINE__, &
                              __FILE__)
            return
        endif
    end subroutine

    module subroutine profiler_set_err_mode(err_mode)
        integer, intent(in) :: err_mode
        if(err_mode == PROFILER_ERR_RETURN &
           .or. err_mode == PROFILER_ERR_FATAL) error_mode = err_mode
    end subroutine

    subroutine find_region(region_name, search_node, region, ierr)
        !! Finds a region given by name in the given path of the profiler
        !! tree. Returns a pointer to that region.
        character(len=*), intent(in) :: region_name
        !! Name of the region
        type(profiler_region_t), target, intent(in) :: search_node
        !! Pointer to the node where to search in for the region
        type(profiler_region_t), pointer, intent(out) :: region
        !! Pointer to the found region
        integer, intent(out) :: ierr
        !! Error code

        integer:: i, ireg
        !! Loop index; region index

        ierr = GENEX_SUCCESS

        ! Check for the case that region is the region tree
        if(region_name == profiler_region_tree%region_name) then
            region => profiler_region_tree
        ! Loop through all children in region path and check if name matches
        else
            ireg = 0
            do i = 1, search_node%number_of_children
                if (region_name == search_node%children(i)%region_name) then
                    ireg = i
                endif
            enddo
            ! Check if region found, otherwise return error
            if(ireg /= 0) then
                region => search_node%children(ireg)
            else
                ierr = GENEX_ERR_PROFILER
            endif
        endif
    end subroutine

    subroutine create_region(region_name, region)
        !! Creates a region at the current path in the profiler tree and
        !! returns a pointer to that region.
        character(len=*), intent(in) :: region_name
        !! Name of the region to create
        type(profiler_region_t), pointer, intent(out) :: region
        !! Pointer to the created region

        call current_node%add_child(region_name)
        region => current_node%children(current_node%number_of_children)

    end subroutine

    subroutine split_path_name(path_name, first, second, back)
        !! Splits the path name by first occurence of the separator "/".
        !! Returns both parts of the split individually. If the path name does
        !! not contain the separator, returns path name in both outputs. If
        !! separator is at the beginning/end, an empty string may be returned
        !! in one of the outputs.
        character(len=*), intent(in) :: path_name
        !! Path name to split
        character(len=*), intent(out) :: first
        !! First part of the split
        character(len=*), intent(out) :: second
        !! Remaining part of the split
        logical, intent(in), optional :: back
        !! If true, split by last occurence of "/" (instead of first).
        !! Default = .false.

        integer :: idx
        !! Index
        logical :: local_back
        !! Local buffer for optional "back" variable

        if(present(back)) then
            local_back = back
        else
            local_back = .false.
        endif

        idx = index(path_name, "/", back=local_back)

        ! If idx = 0, no separator found
        if(idx /= 0) then
            first  = path_name(1 : (idx - 1))
            second = path_name((idx + 1) :)
        else
            first  = path_name
            second = path_name
        endif
    end subroutine

    subroutine find_node(path_name, node, ierr)
        !! Finds a node in the profiler region tree and returns a pointer
        !! to the element given by the node name. Assumes that path_name
        !! contains levels separated by "/" and is an absolute path. Starts
        !! from the root of the tree to find this node.
        character(len=*), intent(in) :: path_name
        !! Name of the path to the node to find
        type(profiler_region_t), pointer, intent(out) :: node
        !! Pointer to the found node
        integer, intent(out) :: ierr
        !! Error code

        character(len=clen_big) :: current_name, remaining_name
        !! Name buffer variables in the search iteration
        type(profiler_region_t), pointer :: current_search_node
        !! Current pointer to the searched node in the search iteration

        ierr = GENEX_SUCCESS

        ! Initialize
        remaining_name = path_name
        current_search_node => profiler_region_tree

        ! Loop which iterates the tree. The idea is to split the path name
        ! sequentially in each iteration until no separator exists. After
        ! each split we go one level deeper into the tree, i.e. update
        ! current_search_node. If we cannot find the next level we terminate.
        ! If the last level is not the name of the current search node, the
        ! find_region method will not find the name and we exit with an error.
        do
            ! If path found, exit.
            if(remaining_name == current_search_node%region_name) then
                node => current_search_node
                ierr = GENEX_SUCCESS
                exit
            ! Otherwise, split remaining_name into top level (current_name) and
            ! the rest (new remaining_name). Initialize the new iteration
            ! starting from the region current_name. If this does not exist
            ! terminate.
            else
                call split_path_name(remaining_name, current_name, &
                                     remaining_name)
                call find_region(current_name, current_search_node, node, ierr)
                if(ierr /= GENEX_SUCCESS) exit
                current_search_node => node
            endif
        enddo
    end subroutine

    module subroutine profiler_inject(region_name, ierr, set_path)
        character(len=*), intent(in) :: region_name
        integer, intent(out) :: ierr
        logical, optional, intent(in) :: set_path

        type(profiler_region_t), pointer :: region
        !! Pointer to region which will be started
        logical :: set_path_local
        !! Local variable to give optional set_path a default value

        ierr = GENEX_SUCCESS

        ! Check for illegal characters in region name
        if(index(region_name, ".") /= 0) then
            call handle_error("Profiler region name """//region_name &
                              //""" contains illegal character "".""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
            goto 500
        endif
        if(index(region_name, "/") /= 0) then
            call handle_error("Profiler region name """//region_name &
                              //""" contains illegal character ""/""!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
            goto 500
        endif

        ! At first time call, we have to initialize the root of the tree.
        if(.not. is_initialized) then
            call profiler_region_tree%initialize(region_name)
            current_node => profiler_region_tree
            current_path = profiler_region_tree%region_name
            is_initialized = .true.
        endif

        ! Find the performance region
        call find_region(region_name, current_node, region, ierr)
        ! Create region if not exists
        if (ierr /= GENEX_SUCCESS) then
            call create_region(region_name, region)
            ierr = GENEX_SUCCESS
        endif

        ! Inject the profiler region
        call region%inject(ierr)

500     if (ierr /= GENEX_SUCCESS) then
            if (error_mode == PROFILER_ERR_RETURN) ierr = GENEX_WRN_PROFILER
            call handle_error("Profiler inject failed!", ierr, __LINE__, &
                              __FILE__)
            return
        endif

        ! Set current path to started region if present
        set_path_local = .false.
        if (present(set_path)) set_path_local = set_path
        if (set_path_local) call profiler_set_path("./"//region_name, ierr)
    end subroutine profiler_inject

    module subroutine profiler_start_allreduce(comm, ierr)
        integer, intent(in) :: comm
        integer, intent(out) :: ierr

        ! Isolate the synchronization time from the communication time
        ! of MPI Allreduce
        if(get_isolate_tsync()) then
            call profiler_start("mpi_barrier", ierr)
            call MPI_Barrier(comm, ierr)
            call profiler_stop("mpi_barrier", ierr)
        endif

        call profiler_start("mpi_allreduce", ierr)
    end subroutine profiler_start_allreduce

    module subroutine profiler_stop_allreduce(ierr)
        integer, intent(out) :: ierr

        ! NOTE: This subroutine currently does not add any functionality
        !       on top of profiler stop but may do so in the future.
        call profiler_stop("mpi_allreduce", ierr)
    end subroutine profiler_stop_allreduce

    module subroutine profiler_inject_allreduce(ierr)
        integer, intent(out) :: ierr

        ! Inject the isolated synchronization region from the communication
        ! of MPI Allreduce
        if(get_isolate_tsync()) then
            call profiler_inject("mpi_barrier", ierr)
        endif

        call profiler_inject("mpi_allreduce", ierr)
    end subroutine profiler_inject_allreduce

end submodule profiler_s
