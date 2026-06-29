submodule (profiler_m) profiler_region_s
    !! Contains the implementation of the profiler_region_t type
    use type_converters_m, only: string_f2c
    use params_gpu_offload_m, only: get_use_gpu_offload
    use runtime_tracer_m, only: trace_profile, RT_TRACER_START, RT_TRACER_END
#ifdef ENABLE_GPU
    use profiler_gpu_m, only: cbind_annotation_start, cbind_annotation_end, &
                              cbind_get_profiler_region_time
#endif

    implicit none

contains

    module subroutine profiler_region_initialize(this, region_name)
        class(profiler_region_t), intent(inout) :: this
        character(len=*), intent(in) :: region_name

        this%number_of_children = 0
        this%number_of_calls    = 0
        this%time               = 0.0_DP
        this%activated          = .false.

        this%region_name = region_name
        call string_f2c(this%region_name, this%region_name_c)
    end subroutine

    module subroutine profiler_region_add_child(this, child_name)
        class(profiler_region_t), intent(inout) :: this
        character(len=*), intent(in) :: child_name

        ! First time allocation of children array
        if(.not. allocated(this%children)) then
            this%current_nmax_children = default_n_regions
            allocate(this%children(this%current_nmax_children))
        endif

        ! Check if maximum number of children reached. Extend array if yes.
        if (this%number_of_children == this%current_nmax_children) then
            call extend_children(this)
        endif

        ! Add children and increase counter
        this%number_of_children = this%number_of_children + 1
        call this%children(this%number_of_children)%initialize(child_name)
    end subroutine

    module subroutine profiler_region_start(this, ierr)
        class(profiler_region_t), intent(inout) :: this
        integer, intent(out) :: ierr

        ! Activate performance region
        if (this%activated) then
            call handle_error("Profiler region """//trim(this%region_name) &
                              //""" already activated!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
        else
            this%activated = .true.
            ierr = GENEX_SUCCESS
        endif

#ifdef ENABLE_GPU
        call cbind_annotation_start(this%region_name_c)
#endif

        this%number_of_calls = this%number_of_calls + 1
        this%time_start      = MPI_wtime()

        call trace_profile(RT_TRACER_START, &
                           trim(current_path), trim(this%region_name), &
                           this%time_start)
    end subroutine

    module subroutine profiler_region_stop(this, ierr)
        class(profiler_region_t), intent(inout) :: this
        integer, intent(out) :: ierr

        real(kind=DP) :: time_end

        ! De-activate performance region
        if (.not. this%activated) then
            call handle_error("Profiler region """//trim(this%region_name) &
                              //""" not activated!", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
            ierr = GENEX_ERR_PROFILER
        else
            this%activated = .false.
            ierr = GENEX_SUCCESS
        endif
        time_end = MPI_wtime()
        this%time = this%time + time_end - this%time_start

        call trace_profile(RT_TRACER_END, &
                           trim(current_path), trim(this%region_name), &
                           time_end)

#ifdef ENABLE_GPU
        call cbind_annotation_end(this%region_name_c)
#endif

        ! Save the activation time for the very first call
        if(this%number_of_calls == 1) then
            this%time_first_call = this%time
        endif
    end subroutine

    module subroutine profiler_region_exchange(this, dcomm_handler, &
                            discard_first, global_calls, global_time, ierr)
        class(profiler_region_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        logical, intent(in) :: discard_first
        integer, intent(out) :: global_calls
        real(kind = DP), intent(out) :: global_time
        integer, intent(out) :: ierr

        integer :: calls
        !! Number of calls to communicate
        real(kind = DP) :: time
        !! Time to communicate

        ! If option discard first is enabled, remove the first call from
        ! the variables to communicate
        if(discard_first .and. (this%number_of_calls > 1)) then

            calls = this%number_of_calls - 1
            time  = this%time - this%time_first_call
        else
            calls = this%number_of_calls
            time  = this%time
        endif

        ! Get the global number of calls and the time from all MPI processes
        call MPI_Reduce(calls, global_calls, 1, MPI_INTEGER, MPI_SUM, 0, &
                        dcomm_handler%get_comm_cart(), ierr)
        call MPI_Reduce(time, global_time, 1, MPI_DOUBLE, MPI_SUM, 0, &
                        dcomm_handler%get_comm_cart(), ierr)
    end subroutine

    recursive module subroutine profiler_region_print(this, dcomm_handler, &
                                    channel, reference_time, lvl, &
                                    discard_first, ierr)
        class(profiler_region_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        integer, intent(in) :: channel
        real(kind = DP), intent(in) :: reference_time
        integer, intent(in) :: lvl
        logical, intent(in) :: discard_first
        integer, intent(out) :: ierr

        integer :: i
        !! Loop index
        integer :: global_calls_to_print
        !! Number of calls from all parallelized processes
        real(kind = DP) :: global_time_to_print
        !! Total time from all parallelized processes

        character(len=8) :: char_calls_global
        !! Character buffer for the number of global calls
        character(len=clen) :: name_fmt
        !! Formated name to print out

        call this%exchange(dcomm_handler, discard_first, &
                           global_calls_to_print, global_time_to_print, ierr)
        if (ierr /= GENEX_SUCCESS) return

        ! Print performance summary
        if(global_calls_to_print > 0) then

            ! NOTE: We write profiler_calls_global into a character
            !       buffer because for debug configuration an error
            !       occurs if the integer is too long for the
            !       specified amount of characters.
            if(global_calls_to_print < 1e8_DP) then
                write(char_calls_global, '(I8)') global_calls_to_print
            else
                char_calls_global = '********'
            end if

            ! Format the name correctly. For each level in the tree we add
            ! 4 spaces to the name in front
            name_fmt = trim(this%region_name)
            do i = 1, lvl
                name_fmt = "  " // name_fmt
            enddo

            ! Print performance summary of this region
            if(dcomm_handler%is_master()) then
                write(channel, &
                      "(X, A37,' | ',A8,' | ',F14.5,' | ',F14.5,' | ',F14.8)") &
                      name_fmt, &
                      char_calls_global, &
                      global_time_to_print, &
                      global_time_to_print / global_calls_to_print, &
                      global_time_to_print / reference_time * 1e2_DP
            endif
        endif

        ! Recursive call to all children to print their summary
        do i = 1, this%number_of_children
            call this%children(i)%print(dcomm_handler, channel, reference_time,&
                                        lvl=lvl+1, &
                                        discard_first=discard_first, &
                                        ierr=ierr)
        enddo

    end subroutine

    recursive module subroutine profiler_region_reset(this)
        class(profiler_region_t), intent(inout) :: this

        integer :: i

        do i = 1, this%number_of_children
            call this%children(i)%reset()
        enddo

        if (allocated(this%children)) then
            deallocate(this%children)
        endif

        call this%initialize('')
    end subroutine

    subroutine extend_children(this)
        !! Extends the children array of a profiler region by a factor of 2
        !! to store more children.
        type(profiler_region_t), intent(inout) :: this
        !! Profiler region which children should be extended

        integer :: i
        !! Loop index
        type(profiler_region_t), dimension(:), allocatable :: buffer
        !! Buffer to save children

        ! NOTE: We make use of our custom subroutine "deepcopy_region"
        !       because with the current version of the gnu compiler we
        !       get a memory leak because the children array and arrays
        !       therein are only shallow-copied.

        ! Deepcopy children array to a buffer and clear children
        allocate(buffer(this%number_of_children))
        do i = 1, this%number_of_children
            call deepcopy_region(from=this%children(i), to=buffer(i))
        enddo
        deallocate(this%children)

        ! Extend the maximum number of children by a factor of 2 and allocate a
        ! new, bigger children array
        this%current_nmax_children = 2 * this%current_nmax_children
        allocate(this%children(this%current_nmax_children))

        ! Deepcopy elements from buffer to children array
        do i = 1, this%number_of_children
            call deepcopy_region(from=buffer(i), to=this%children(i))
        enddo
        deallocate(buffer)

    end subroutine

    recursive subroutine deepcopy_region(from, to)
        !! Deep-copies the profiler region "from" to the region "to". This
        !! means that the complete memory of the "from" region is replicated
        !! and assigned to the "to" region. The children array will be
        !! deep-copied as well by recursive call of this subroutine.
        type(profiler_region_t), intent(inout) :: from
        !! Source region from where to take the data
        type(profiler_region_t), intent(inout) :: to
        !! Destination region where the contents of source should be copied to

        integer :: i
        !! Loop index

        ! NOTE: We can not simply use "to = from" because of the allocatable
        !       children array. We have to copy the components manually and
        !       also treat the children manually.

        ! Directly copy the intrinsic, non-allocatable components
        to%number_of_children    = from%number_of_children
        to%number_of_calls       = from%number_of_calls
        to%time                  = from%time
        to%time_start            = from%time_start
        to%time_first_call       = from%time_first_call
        to%activated             = from%activated
        to%region_name           = from%region_name
        to%region_name_c         = from%region_name_c
        to%current_nmax_children = from%current_nmax_children

        ! If children exist, allocate the new children array in the destination
        ! and recursively call the deepcopy subroutine on the children.
        if(to%number_of_children > 0) then
            allocate(to%children(to%current_nmax_children))
            do i = 1, to%number_of_children
                call deepcopy_region(from%children(i), to%children(i))
            enddo
        endif
    end subroutine

    module subroutine profiler_region_inject(this, ierr)
        class(profiler_region_t), intent(inout) :: this
        integer, intent(out) :: ierr

        real(kind=DP) :: times(2)
        integer :: n_calls(1)
        integer :: ierr_cxx = 1
        logical :: is_first_time_call

        ! Check if this is the first call
        is_first_time_call = .false.
        if(this%number_of_calls == 0) then
            is_first_time_call = .true.
        endif

        ! Fetch the time measurement from the C++ profiler
#ifdef ENABLE_GPU
        if(get_use_gpu_offload()) then
            ierr_cxx = cbind_get_profiler_region_time(this%region_name_c, &
                                                      times, n_calls)
        else
            call handle_error("Illegal call of profiler injection! &
                              &(C++ layer is not used)", &
                              GENEX_WRN_PROFILER, __LINE__, __FILE__)
        endif
#else
        call handle_error("Illegal call of profiler injection! &
                          &(C++ layer was not built)", &
                          GENEX_WRN_PROFILER, __LINE__, __FILE__)
#endif
        if(ierr_cxx /= 0) then
            ierr = GENEX_ERR_PROFILER
        else
            ierr = GENEX_SUCCESS
        endif

        ! Store the number of calls fetched from C++
        this%number_of_calls = this%number_of_calls + n_calls(1)

        ! Store the measured time interval fetched from C++
        this%time = this%time + times(1)

        ! Save the activation time for the very first call
        if(is_first_time_call) then
            this%time_first_call = times(2)
        endif
    end subroutine

end submodule profiler_region_s
