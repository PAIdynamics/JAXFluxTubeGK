module profiler_m
    !! Module to analyse the runtime of annotated regions of the program. A
    !! region is selected for time measurements by surrounding it with calls
    !! to profiler_start and profiler_stop. A performance summary of the
    !! different regions is printed on screen by calling profiler_print.
    use MPI
    use, intrinsic :: iso_c_binding, only: C_CHAR
    use genex_fortran_env_m, only: DP
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only : GENEX_SUCCESS, GENEX_ERR_PROFILER, &
                                     GENEX_WRN_PROFILER
    use dcomm_handler_m, only : dcomm_handler_t

    implicit none
    private

    ! This module contains the profiler, a state machine that internally uses
    ! a tree structure of type profiler_region_t for profiling of the code.
    ! The variable current_node sets a pointer to a specific part in the
    ! tree that is then used as a base for all following operations such as
    ! start, stop and print. Upon start of a profiler region, it will be
    ! automatically created if it does not exist. The location of the creation
    ! will be the current node. Upon stop, the profiler will try to find
    ! the region to stop in the current node. For print, the reference region
    ! will be searched in the current node, whereas the print itself will
    ! recursively print the performance summary of the whole tree.

    ! Terminology:
    ! Region ... A profiling region which runtime should be monitored
    ! Tree   ... Recursive structure that contains all profiling regions
    ! Root   ... The first (top level) element of the tree
    ! Node   ... A single region in the tree (like a branch fork in a tree)
    ! Path   ... String representation of a sequence of nodes, i.e. a path to
    !            a specific node. Separates nodes with special character "/".

    integer, parameter :: clen = 38
    !! Maximum character length of printed strings
    integer, parameter :: clen_big = 128
    !! Maximum character length of strings specifying tree locations
    integer, parameter :: default_n_regions = 10
    !! Default number of performance regions within a level of the tree

    type, private :: profiler_region_t
        !! Type that represents a performance region. Contains the name
        !! as well as the measured time and amount of calls. Can be used to
        !! start and stop performance measurements and to print a summary.
        !! Contains an array of children performance regions.
        private
        character(len=clen) :: region_name
        !! Profiler region name in Fortran character format
        character(len=1, kind=C_CHAR), dimension(clen) :: region_name_c
        !! Profiler region name in C character array format
        real(kind = DP) :: time_first_call
        !! Wtime of the first call
        real(kind = DP) :: time_start
        !! Wtime of the last start of performance monitoring
        real(kind = DP) :: time
        !! Total measured and accumulated runtime
        logical :: activated
        !! True if performance measuring is activated
        integer :: number_of_calls
        !! Number of calls of the performance measuring
        integer :: number_of_children
        !! Number of performance region children inside this type
        integer :: current_nmax_children
        !! Current maximum number of children that fit into the children array
        type(profiler_region_t), dimension(:), allocatable :: children
        !! Array with performance region children inside this type
    contains
        procedure :: initialize => profiler_region_initialize
        procedure :: add_child  => profiler_region_add_child
        procedure :: start      => profiler_region_start
        procedure :: stop       => profiler_region_stop
        procedure :: exchange   => profiler_region_exchange
        procedure :: print      => profiler_region_print
        procedure :: reset      => profiler_region_reset
        procedure :: inject     => profiler_region_inject
    end type

    interface
        module subroutine profiler_region_initialize(this, region_name)
            !! Initializes the profiler region type
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            character(len=*), intent(in) :: region_name
            !! Name of the profiler region
        end subroutine

        module subroutine profiler_region_add_child(this, child_name)
            !! Adds a child to the profiler region. Allocates additional
            !! memory for more children to be added if current children
            !! array becomes too small.
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            character(len=*), intent(in) :: child_name
            !! Name of the child to add
        end subroutine

        module subroutine profiler_region_start(this, ierr)
            !! Starts the time measurement of the profiler region
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine profiler_region_stop(this, ierr)
            !! Stops the time measurement of the profiler region
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine profiler_region_exchange(this, dcomm_handler, &
                                discard_first, global_calls, global_time, ierr)
            !! Exchanges the profiling information across all processes in
            !! the parallelized domain.
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler
            logical, intent(in) :: discard_first
            !! If true, the first call will be discarded from the measurement
            integer, intent(out) :: global_calls
            !! Number of calls of all parallelized instances
            real(kind = DP), intent(out) :: global_time
            !! Total time of all parallelized instances
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        recursive module subroutine profiler_region_print(this, dcomm_handler, &
                                        channel, reference_time, lvl, &
                                        discard_first, ierr)
            !! Prints a performance summary of the global amount of calls and
            !! the total time in all processes across the parallelized domain.
            !! Recursively calls a print to all children.
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler
            integer, intent(in) :: channel
            !! Output channel to print performance summary
            real(kind = DP), intent(in) :: reference_time
            !! Reference time to print relative execution times
            integer, intent(in) :: lvl
            !! Current level of the recursive call. Determines the name format
            !! with preceeding spaces according to the level.
            logical, intent(in) :: discard_first
            !! If true, the first call will be discarded from the measurement.
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        recursive module subroutine profiler_region_reset(this)
            !! Resets the profiler region type
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine profiler_region_inject(this, ierr)
            !! Inject a profiler region to the profiler after the event with
            !! time measurements fetched from the C++ profiler. This can only be
            !! called if the C++ layer is built and used.
            class(profiler_region_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: ierr
            !! Error code
        end subroutine
    end interface

    logical :: is_initialized = .false.
    !! Bookkeeping variable that is used to initialize the tree. Upon first
    !! call of profiler_start, the variable profiler_region_tree will be
    !! initialized with the given region name.
    type(profiler_region_t), target :: profiler_region_tree
    !! Root of the tree of profiler regions
    type(profiler_region_t), pointer :: current_node
    !! Current tree node used for stop and start of profiling regions
    character(len=clen_big) :: current_path
    !! Path to current node. Saved to back in relative mode of profiler_set_path

    public :: PROFILER_ERR_RETURN, PROFILER_ERR_FATAL
    enum, bind(C)
    !! Enumerator defining the error codes
        enumerator :: PROFILER_ERR_RETURN = 0
        enumerator :: PROFILER_ERR_FATAL
    end enum

    integer :: error_mode = PROFILER_ERR_RETURN
    !! Internal variable to keep track of the current error mode chosen

    public :: profiler_reset
    public :: profiler_start
    public :: profiler_stop
    public :: profiler_print
    public :: profiler_set_path
    public :: profiler_set_err_mode
    public :: profiler_inject
    public :: profiler_start_allreduce
    public :: profiler_stop_allreduce
    public :: profiler_inject_allreduce

    interface
        module subroutine profiler_reset()
            !! Resets all time measuring regions
        end subroutine

        module subroutine profiler_start(region_name, ierr, set_path)
            !! Starts a time measurement region. Automatically creates a region
            !! in the current path if the region does not exist yet. The first
            !! call to start will initialize the tree structure. The path name
            !! shall not contain any of the special characters "." and "/".
            character(len=*), intent(in) :: region_name
            !! Name of the region
            integer, intent(out) :: ierr
            !! Error code
            logical, optional, intent(in) :: set_path
            !! If true set the current profiler path to the started region
            !! (default false)
        end subroutine

        module subroutine profiler_stop(region_name, ierr, set_path)
            !! Stops a time measurement region. Needs the current path to be
            !! correctly set such that it contains the region that should be
            !! stopped.
            character(len=*), intent(in) :: region_name
            !! Name of the region
            integer, intent(out) :: ierr
            !! Error code
            logical, optional, intent(in) :: set_path
            !! If true set the current profiler path to the region above
            !! (default false)
        end subroutine

        module subroutine profiler_print(dcomm_handler, channel, &
                                         ref_region_name, ierr, &
                                         discard_first_call)
            !! Prints the accumulated measurement results over multiple calls
            !! in the given IO channel. Results are summed over multiple MPI
            !! processes and normalized with respect to a reference region.
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler
            integer, intent(in) :: channel
            !! Fortran IO unit where the measurement results are printed to
            character(len=*), intent(in) :: ref_region_name
            !! Name of the reference region
            integer, intent(out) :: ierr
            !! Error code
            logical, optional, intent(in) :: discard_first_call
            !! True if the first measurement should be discarded in the
            !! performance analysis, false otherwise.
        end subroutine

        module subroutine profiler_set_path(path_name, ierr)
            !! Sets the current node pointer to node name given as a path
            !! string. When calling without initialization via start,
            !! the subroutine will initialize the tree with the path name
            !! but only if no separators are present in the name, i.e.
            !! it specifies the top node. Supports two operational modes.
            !! Absolute mode:
            !!   The name should be given as an absolute path where
            !!   the individual levels are separated by a "/".
            !! Relative mode:
            !!   The name can be given in the format "./region_name", where
            !!   the new path will point to the node given by region_name which
            !!   is inside the current path.
            !!   The name can be given by "../" to go one level back in the
            !!   tree. I.e. to the parent region of the current path.
            character(len=*), intent(in) :: path_name
            !! Path name to set the current path to.
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine profiler_set_err_mode(err_mode)
            !! Sets the way the profiler responds to internal errors due to
            !! incorrect use. When set to errors return, the profiler returns
            !! an error code without terminating, expecting that the error
            !! will be handled by the calling routine. When set to errors are
            !! fatal, the code will be interrupted.
            integer, intent(in) :: err_mode
            !! Error mode: use PROFILER_ERR_RETURN or PROFILER_ERR_FATAL. Other
            !!             values are ignored.
        end subroutine

        module subroutine profiler_inject(region_name, ierr, set_path)
            !! Injects a profiler region to the profiler tree after the event
            !! with time measurements fetched from the C++ profiler. The routine
            !! automatically creates a region in the current path if the region
            !! does not exist yet. The first call of the routine will initialize
            !! the tree structure. This can only be called if the C++ layer is
            !! built and used.
            character(len=*), intent(in) :: region_name
            !! Name of the region
            integer, intent(out) :: ierr
            !! Error code
            logical, optional, intent(in) :: set_path
            !! If true set the current profiler path to the started region
            !! (default false)
        end subroutine

        module subroutine profiler_start_allreduce(comm, ierr)
            !! Starts a time measurement region for isolating synchronization
            !! time in MPI_Allreduce communication region
            integer, intent(in) :: comm
            !! MPI_Allreduce communicator
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine profiler_stop_allreduce(ierr)
            !! Stops a time measurement region for isolating synchronization
            !! time in MPI_Allreduce communication region
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine profiler_inject_allreduce(ierr)
            !! Injects an additional profiler region for the isolated
            !! synchronization time in MPI_Allreduce communication region
            integer, intent(out) :: ierr
            !! Error code
        end subroutine
    end interface
end module profiler_m
