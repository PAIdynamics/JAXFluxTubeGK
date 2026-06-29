!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! $Id: grid.F90 1005 2009-07-02 16:12:03Z  $
!> Provide the local grid sizes and associated quantities.
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
module grid

  use global
  use general, only : gkw_warn

  implicit none

  private
  
  !
  ! public procedures
  !
  
  public :: grid_read_nml, grid_bcast_nml, grid_check_params, grid_write_nml
  public :: setup_grid
  
  !
  ! public variables
  ! 
   
  ! the *global* sizes of the grids (only for parallelizable directions)
  integer, public :: number_of_species !< total number of species
                                       ! (N.B. do not count the adiabatic species) 
  integer, public :: n_mu_grid         !< total number of magnetic moment grid points
  integer, public :: n_vpar_grid       !< total number of vparallel grid points
  integer, public :: n_s_grid          !< total number of s grid points
  integer, public :: n_x_grid          !< total number of x grid points
  
  ! the sizes of the *local* grids (per processor)
  ! These are derived from the global sizes in setup_grid
  integer, public :: ns    !< number of grid points along the field line
  integer, public :: nmu   !< number of grid points in the magnetic moment direction
  integer, public :: nvpar !< number of grid points for the parallel velocity 
  integer, public :: nx    !< number of radial wave vectors
  integer, public :: nmod  !< total number of toroidal modes
  integer, public :: nsp   !< the number of species
  
  !> value specifies the length of the field line, i.e. the field line makes 
  !> 2*nperiod - 1 poloidal turns. For nonlinear runs nperiod should be 1. 
  integer, public :: nperiod

  !> The number of grid points in the trapping region with positive parallel
  !> velocity (used when vp_trap = 1). The total number in the trapped domain
  !> is 2*n_trapped.
  integer, public :: n_trapped

  !>The maximum values of the velocity grids
  real, public :: vpmax=3.E0
  real, public :: mumax=3**2/2.E0

  ! The number of processors to be used in each parallel direction;
  ! these are either set automatically (if possible) in setup_grid 
  ! (based on the total number of processors), or are prescribed via the
  ! GRIDSIZE namelist (previously in CONTROL).
  integer, public :: n_procs_s        !< # of procs to divide up the s-direction
  integer, public :: n_procs_mu       !< # of procs to divide up the mu grid
  integer, public :: n_procs_sp       !< # of procs to divide up the species direction
  integer, public :: n_procs_vpar     !< # of procs to divide up the parallel velocity

  ! Logicals to know if we are parallel over some dimension; these are
  ! set in setup_grid
  logical, public :: parallel_sp       !< True if the code parallelizes over species 
  logical, public :: parallel_mu       !< True if the code parallelizes over mu grid
  logical, public :: parallel_s        !< True if the code parallelizes over s 
  logical, public :: parallel_vpar     !< True if the code parallelizes over vpar grid

  !> rank of the next processor in the vpar direction 
  integer, public :: proc_vpar_next
  !> rank of the previous processor in the vpar direction 
  integer, public :: proc_vpar_prev
  !> rank of the next processor in the mu direction 
  integer, public :: proc_mu_next
  !> rank of the previous processor in the mu direction 
  integer, public :: proc_mu_prev
  !> rank of the next processor in the s direction 
  integer, public :: proc_s_next
  !> rank of the previous processor in the s direction 
  integer, public :: proc_s_prev
  !> rank of the next processor in the vpar and mu 
  integer, public :: proc_vpar_next_mu_next
  !> rank of the previous processor in the vpar direction and next in mu
  integer, public :: proc_vpar_prev_mu_next
  !> rank of the next processor in the vpar and prev in mu
  integer, public :: proc_vpar_next_mu_prev
  !> rank of the previous processor in the vpar direction and prev in mu
  integer, public :: proc_vpar_prev_mu_prev
 
  !> local processor is at lower boundary in vpar
  logical, public :: lproc_vpar_lowerb
  !> local processor is at upper boundary in vpar
  logical, public :: lproc_vpar_upperb
  !> local processor is at lower boundary in s
  logical, public :: lproc_s_lowerb
  !> local processor is at upper boundary in s
  logical, public :: lproc_s_upperb
  !> local processor is at lower boundary in mu
  logical, public :: lproc_mu_upperb
  !> local processor is at lower boundary in mu
  logical, public :: lproc_mu_lowerb
  
  !> send and recv boundary in the s direction
  logical, public :: lsendrecv_s = .false.
  !> send and recv boundary in the vpar direction
  logical, public :: lsendrecv_vpar = .false.
  !> send and recv boundary in the mu direction (used with e.g. collisions)
  logical, public :: lsendrecv_mu = .false.

  !> ixpb contains the first x point on the local processor
  integer, public :: ixpb
  !> isppb contains the first of the species number on the local processor
  integer, public :: isppb
  !> isppe ccontains the last of the species number on the local processor
  integer, public :: isppe
  !> imupb contains the first mu grid point on the local processor
  integer, public :: imupb
  !> imupe contains the last mu grid point on the local processor
  integer, public :: imupe
  !> ispb contains the first s grid point on the local processor
  integer, public :: ispb
  !> ispe contains the last s grid point on the local processor
  integer, public :: ispe
  !> ivparpb contains the first vpar grid point on the local processor.
  !> For vp_trap = 1, the processor will be responsible for a mirror point
  !> for every point apparently in the grid.
  integer, public :: ivparpb
  !> ivparpe contains the first vpar grid point on the local processor.
  !> For vp_trap = 1, the processor will be responsible for a mirror point 
  !> for every point apparently in the grid.
  integer, public :: ivparpe
  
  !> position of the processor in the s direction (1 - n_procs_s)
  integer, public :: iproc_s
  
  !> *request* non-blocking boundary sendrecv in vpar direction
  logical, public :: non_blocking_vpar

  !
  ! locals
  !
  
  !> number of possible parallel directions
  integer, parameter :: nplan = 4
  !> plan ordering
  integer, parameter :: isp = 1, imu = 2, ivpar = 3, is = 4
 
  !> Convenient way of dealing with the plan in this module only.
  !> Each plan corresponds to a parallel direction.
  type :: plan
    logical :: parallel = .false.      !< true if parallel
    integer :: procs    = 0            !< number of procs
    integer :: points   = 0            !< number of points per proc
    integer :: min_points   = 0        !< minimum number of points per proc
    integer :: max_points   = 0        !< max. no. of points per proc in parallel
    character (len=8) :: name = 'none' !< name
    integer :: ipb                     !< first point on this proc 
    integer :: ipe                     !< last point on this proc
    integer :: iproc                   !< processor in the ? direction
    logical :: cart = .false.          !< if the direction needs local comm
    logical :: periodic = .false.      !< if the direction is periodic
    integer :: direction = -1          !< direction for MPI calls in cartesian
  end type plan 
  
  !> structure to hold parallel plan details and associated indicies
  type (plan), save, dimension(nplan) :: pp !< pp is the plan structure

  !> re-order allowed in creating sub-communicators
  logical, parameter :: reorder = .true.

  ! other locals
  integer :: n_dims, n_cart_dims
  integer :: parallel_dim_count, i, j
  integer :: ierr = 0, iproc, k, l
  logical, dimension (nplan) :: periodic
  integer, dimension (nplan) :: dims,coords

  !
  ! interfaces
  !
  
  ! use the same routine for reading and writing
  interface grid_write_nml
    module procedure grid_read_nml
  end interface

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> In this subroutine it is determined how the calculation is divided
!> among the different processors. This can be done automatically, by
!> setting all the processor numbers to 1 in the input file (or not 
!> including them) or manually, by specifying n_procs_[mu|sp|s|vpar] in
!> the control namelist. Manual setup is implied if *any* of n_procs_*
!> are found to be > 1. N.B. this routine MUST return nsp,ns,nmu and
!> nvpar (the LOCAL processor problem sizes). 
!----------------------------------------------------------------------------
subroutine setup_grid

  use mpiinterface
  use control, only : order_of_the_scheme, vp_trap, collisions

  integer :: ilun

  ! make nothing parallel by default
  pp(:)%parallel   = .false.

  ! put some value in by default
  pp(:)%max_points = i_huge
  
  !
  ! put the input parameters into the plan structure
  !
  
  ! species:
  pp(isp)%name       = 'species'
  pp(isp)%points     = number_of_species
  pp(isp)%procs      = n_procs_sp
  pp(isp)%min_points = 1

  ! mu grid:
  pp(imu)%name       = 'mu'
  pp(imu)%points     = n_mu_grid
  pp(imu)%procs      = n_procs_mu
  pp(imu)%min_points = 1
  if (collisions) pp(imu)%min_points = 2

  ! vpar grid:
  pp(ivpar)%name       ='vpar'
  pp(ivpar)%points     = n_vpar_grid
  pp(ivpar)%procs      = n_procs_vpar
  if (vp_trap == 1) then
    pp(ivpar)%min_points = n_vpar_grid
  else
    select case(order_of_the_scheme)
      case('second_order')
        pp(ivpar)%min_points = 1 
        pp(ivpar)%max_points = n_vpar_grid 
      case('fourth_order')
        pp(ivpar)%min_points = 2 
        pp(ivpar)%max_points = n_vpar_grid 
      case default
        pp(ivpar)%min_points = i_huge
    end select
  end if

  ! s grid:
  pp(is)%name       ='s' 
  pp(is)%points     = n_s_grid
  pp(is)%procs      = n_procs_s 
  pp(is)%min_points = 2

  !write(*,*) pp(:)%procs
  ! Use the above to determine the number of processors to be used in each
  ! direction.
  call parallel_plan

  ! Copy the number of processsors obtained into the variables use by the
  ! rest of the code and check the layout before doing any more.
  n_procs_s   = pp(is)%procs
  n_procs_vpar= pp(ivpar)%procs
  n_procs_mu  = pp(imu)%procs
  n_procs_sp  = pp(isp)%procs

  ! set nsp, nmu, nvpar, ns
  nsp   = pp(isp)%points
  nmu   = pp(imu)%points
  nvpar = pp(ivpar)%points
  ns    = pp(is)%points

  ! set parallel_* logicals
  parallel_sp      = pp(isp)%parallel
  parallel_mu      = pp(imu)%parallel
  parallel_vpar    = pp(ivpar)%parallel
  parallel_s       = pp(is)%parallel

  !Also check allowed inputs with parallel options
  !Do not move above logicals above.
  call check_parallel_layout

  ! set non_blocking_vpar
  non_blocking_vpar = non_blocking_vpar .and. parallel_vpar .and. &
                    & (vp_trap == 0)

  ! enable sendrecv dist in the mu-direction with collisions
  lsendrecv_mu = parallel_mu .and. collisions
  lsendrecv_s  = parallel_s
  lsendrecv_vpar = parallel_vpar .and. (vp_trap == 0)
  
  ! Configure the processors for nearest neighbour communication of the
  ! boundaries in some of the parallel directions and determine the first
  ! and last points the local processor is responsible for.
  call setup_proc_coords
  call setup_communicators
  call get_local_ranks
  call setup_local_grid_ranges
  call openmpi_fix

  ! explain (via stdout) how things will be parallelized
  if (number_of_processors > 1) call parallel_report(pp)

end subroutine setup_grid 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Read the gridsize namelist
!----------------------------------------------------------------------------

subroutine grid_read_nml(ifile,io_stat,lwrite)

  integer, intent(in)  :: ifile
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /gridsize/ nx, nperiod, n_vpar_grid, nmod, number_of_species,     &
                    & n_mu_grid, n_trapped, n_s_grid, n_procs_s, n_procs_sp, &
                    & n_procs_vpar, n_procs_mu, non_blocking_vpar, vpmax, mumax


  io_stat = 0 

  ! read nml, if not writing
  if (present(lwrite)) then
    
    if (.not. lwrite) then
    
    ! Set the default sizes of the grid to zero
    number_of_species = 0 
    n_mu_grid         = 0
    n_vpar_grid       = 0
    n_s_grid          = 0
    
    ! Set other default values
    nperiod           = 1
    n_trapped         = 0
    nx                = 1
    nmod              = 1

    ! import any values set in control for the defaults
    n_procs_s         = 1
    n_procs_vpar      = 1
    n_procs_mu        = 1
    n_procs_sp        = 1
    non_blocking_vpar = .true.
 
    ! read nml
    read(ifile,NML=gridsize,IOSTAT=io_stat)
    else
      ! do nothing
    end if
    
  else

    ! write nml
    write(ifile,NML=gridsize)

  end if
  
end subroutine grid_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> distribute the (global) grid sizes over the different processors
!----------------------------------------------------------------------------

subroutine grid_bcast_nml

  use mpiinterface

  call mpibcast_integer(number_of_species,1)
  call mpibcast_integer(n_mu_grid,        1)
  call mpibcast_integer(n_vpar_grid,      1)
  call mpibcast_integer(n_s_grid,         1)
  call mpibcast_integer(nx,               1)
  call mpibcast_integer(nmod,             1)
  call mpibcast_integer(n_trapped,        1)
  call mpibcast_integer(nperiod,          1)
  call mpibcast_integer(n_procs_s,        1)
  call mpibcast_integer(n_procs_vpar,     1)
  call mpibcast_integer(n_procs_mu,       1)
  call mpibcast_integer(n_procs_sp,       1)
  call mpibcast_logical(non_blocking_vpar,1)

end subroutine grid_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> perform checks on the grid params
!----------------------------------------------------------------------------

subroutine grid_check_params
 
  use general, only : gkw_abort, gkw_warn
  use control, only : vp_trap, collisions

  ! n_x_grid = nx for now
  n_x_grid = nx
  
  ! check we have positive values for the n_procs_*
  if (n_procs_s <= 0 .or. n_procs_mu <= 0 .or. n_procs_sp <= 0 .or.          &
      &                                            n_procs_vpar <= 0) then
    call gkw_abort('control: '//                                             &
        &          'n_procs_[s|mu|sp|vpar] <= 0 not allowed!')
  endif
  
  ! check for non_blocking_vpar without vp_trap = 0
  if (non_blocking_vpar .and. (vp_trap /= 0)) then
    call gkw_warn('control: non_blocking_vpar selected without '//           &
        &         'vp_trap = 0')
  endif
 
  ! nperiod:
  if (nperiod <= 0) then
    call gkw_abort('grid_size: unreasonable value of nperiod')   
  end if
  
  ! if vp_trap we require an even number of points in n_vpar_grid
  if (vp_trap == 1 .and. mod(n_vpar_grid,2) /= 0) then
    call gkw_abort('grid_size: '//                                           &
        &          ' for vp_trap we require n_vpar_grid to be even')
  end if

  ! check if the grid sizes are set
  if (number_of_species == 0) then
    call gkw_abort('grid_size: '//                                           &
        &          'The number of species is zero; set "number_of_species".')
  end if
  if (n_mu_grid == 0 .or. n_vpar_grid == 0 .or. n_s_grid == 0) then
    call gkw_abort('grid_size: '//                                           &
        &          'There are no points in one of n_[s|mu|vpar]_grid; '//    &
        &          'check the input file')
  end if
  
end subroutine grid_check_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> find out how many processors we shall use in each direction
!----------------------------------------------------------------------------

subroutine parallel_plan

  use mpiinterface
  use general, only : gkw_warn, gkw_abort 
  use control, only : vp_trap
  use global,  only : lverbose

  integer, dimension(nplan) :: itest, procs
  integer :: i, j, k
  logical :: done

  ! In order to obtain the maximum number of procs in each direction we
  ! must increase the minimum number of points in that direction until
  ! it divides the total number of points (for efficiency).
  ! In the case where we have less points than the minimum, we set the
  ! minimum to the total number of points, then this will get done on 1
  ! processor in this direction.

  do i=1, nplan
    if (pp(i)%points < pp(i)%min_points) then
      pp(i)%min_points = pp(i)%points
      if (lverbose) then
        write(*,*) '* reducing the min_points in ',pp(i)%name,' to',&
                  & pp(i)%points
      end if
    else
      addpoints : do
        if ( mod(pp(i)%points,pp(i)%min_points) == 0 ) exit addpoints
        pp(i)%min_points = pp(i)%min_points + 1
        if (lverbose) then
          write(*,*) '* increasing min_points in ',pp(i)%name,' to',&
                    & pp(i)%min_points
        end if
      end do addpoints
    end if
  end do

  ! check various possibilities
  done = .false.

  ! check for single processor -- override any input number of processors 
  ! and print a warning message
  if (number_of_processors == 1) then
    done = .true.
    if (any(pp(:)%procs > 1)) then
       call gkw_warn('setup_grid:&
                    & n_procs_<something> has been set > 1 for a single&
                    & processor run')
    end if
    pp(:)%procs = 1
    pp(:)%parallel = .false.
  end if

  ! Check if processor numbers have been specified manually (and that they are
  ! consistent with the total number of processors). Avoid the other checks 
  ! below if this is the case.
  if (.not. done) then
    do i=1, nplan

      ! check if the number of processors in one direction has been specified
      if ( pp(i)%procs > 1 ) then
        pp(i)%parallel = .true.

        ! check that this number of processors divides the total
        ! number of points
        if (mod(pp(i)%points,pp(i)%procs) == 0) then
          pp(i)%points = pp(i)%points / pp(i)%procs
        else
          call gkw_abort('setup_grid:&
                        & mismatch of points/procs for '//pp(i)%name)
        end if

        ! check if the specified number is too big
        if (pp(i)%points < pp(i)%min_points) then
          call gkw_abort('setup_grid:'//                                     &
                        &' too few points per proc (or too many processors'//&
                        &' specified ) in '//pp(i)%name)
        end if
        done = .true.
      end if

    end do

  end if

  ! check if we can parallelize over only 1 of species or mu 
  ! (or vpar when vp_trap=1)
  if (.not. done) then
    itest(1)=isp
    call get_parallel_combination(number_of_processors,1,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,1)
  end if
  if (.not. done) then
    itest(1)=imu
    call get_parallel_combination(number_of_processors,1,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,1)
  end if
  if ((.not. done) .and. vp_trap == 1) then
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,1,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,1)
  end if

  ! *both* species and mu (and also vpar when vp_trap=1)
  if (.not. done) then
    itest(2)=isp
    itest(1)=imu
    call get_parallel_combination(number_of_processors,2,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,2)
  end if
  if ((.not. done) .and. vp_trap == 1) then
    itest(2)=isp
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,2,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,2)
  end if
  if ((.not. done) .and. vp_trap == 1) then
    itest(2)=imu
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,2,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,2)
  end if
  
  ! species, mu and vpar (for vp_trap=1)
  if ((.not. done) .and. vp_trap == 1) then
    itest(3)=isp
    itest(2)=imu
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,3,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,3)
  end if

  ! species, mu and vpar (for vp_trap=0)
  if ((.not. done) .and. vp_trap .eq. 0) then
    itest(3)=isp
    itest(2)=imu
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,3,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,3)
  end if

  ! vpar and (species or mu) (for vp_trap=0)
  if ((.not. done) .and. vp_trap .eq. 0) then
    itest(1)=ivpar
    itest(2)=isp
    call get_parallel_combination(number_of_processors,2,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,2)
  end if
  if ((.not. done) .and. vp_trap .eq. 0) then
    itest(1)=ivpar
    itest(2)=imu
    call get_parallel_combination(number_of_processors,2,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,2)
  end if
  
  ! just vpar alone (when vp_trap=0)
  if ((.not. done) .and. vp_trap .eq. 0) then
    itest(1)=ivpar
    call get_parallel_combination(number_of_processors,1,itest,procs,done)
    if (done) call set_parallel_plan(itest,procs,1)
  end if

  ! --- no more combinations needed yet ---

  ! (the below is for any combination of 3)
  !  
  !  if (.not. done) then
  !    three : do i=1, nplan
  !      do j=2, nplan
  !        do k=3, nplan
  !          if (i .ne. j .and. j .ne. k .and. i .ne. j) then
  !            itest(1) = i
  !            itest(2) = j
  !            itest(3) = k
  !            ! don't allow combinations involving "is" or "ivpar" yet!
  !            if( (.not. any(itest(1:3) .eq. is)) .and. vp_trap .eq. 1) then
  !              call get_parallel_combination(number_of_processors, 3, itest, &
  !                                          & procs, done)
  !              if (done) then
  !                call set_parallel_plan(itest, procs, 3)
  !                exit three
  !              end if
  !            end if
  !          end if
  !        end do
  !      end do
  !    end do three
  !  end if
  
  
  ! give up if no possible combinations are found
  if (.not. done) call gkw_abort('parallel_plan:'//                          &
                  &' no processor and gridsize combinations possible for'//  &
                  &' this number of processors')
 
end subroutine parallel_plan

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine checks if it might be possible to parallelize over the
!> first direction contained in itest. Calling recursively tests all the
!> possibilities for the next directions.
!----------------------------------------------------------------------------

recursive subroutine get_parallel_combination(&
                       & total_procs, n, itest, nprocs, found)

  use general, only : gkw_abort 
  use control, only : collisions, ltrapping_arakawa
  use global,  only : lverbose

  !> number of processors available for the decomposition
  integer, intent(in) :: total_procs
  !> number of directions to test
  integer, intent(in) :: n
  !> index of the directions to test
  integer, dimension(nplan), intent(in) :: itest

  !> number of processors in each direction
  integer, dimension(nplan), intent(out) :: nprocs
  !> denotes combination found succesfully
  logical, intent(out) :: found

  !>the maximum number of processors we can use for this combination
  integer :: m_max_procs
  !> maxmimum processors in each direction
  integer, dimension(nplan) :: max_procs
  !> number of points per proc in each direction
  integer, dimension(nplan) :: npoints
  

  ! local variables  
  integer :: i, j, procs_i, procs_remain
  integer, dimension(nplan) :: itest_next
  integer, dimension(nplan) :: nprocs_next
  logical :: found_next, good
  integer :: s_vpar_count,vpar_mu_count
  
  ! defaults
  found = .false.
  found_next = .false.

  !if (lverbose) write (*,*) '* itest: n=',n
  !if (lverbose) write (*,*) '* testing: ',pp(itest(1:n))%name

  s_vpar_count = 0 ; vpar_mu_count = 0
  ! checks - probably not necessary (could just return?) 
  do i=1, n
    if (itest(i) == is) s_vpar_count = s_vpar_count + 1
    if (itest(i) == ivpar ) then
      s_vpar_count = s_vpar_count + 1
      vpar_mu_count = vpar_mu_count + 1
    end if
    if (itest(i) == imu ) vpar_mu_count = vpar_mu_count + 1    
    do j=1, n
      if (i .ne. j) then
        if (itest(i) .eq. itest(j)) then
          call gkw_abort('get_parallel_combination: bad input')
        end if
      end if
    end do
  end do
  
  if (collisions .and. vpar_mu_count >= 2) then
    if (lverbose) write (*,*) '* skipping both ivpar and imu with collisions'
    return
  end if
  if (ltrapping_arakawa .and. s_vpar_count >= 2) then
    if (lverbose) write (*,*) '* skipping both ivpar and is with trapping Arakawa'
    return
  end if

  ! set the max procs in each direction
  do i=1, n
    max_procs(i) = pp(itest(i))%points / pp(itest(i))%min_points
  end do

  ! set the maximum number of procs we can possibly use
  m_max_procs = 1
  do i=1, n
    m_max_procs = m_max_procs * max_procs(i)
  end do
  
  ! check if we have too many processors
  if (total_procs > m_max_procs) then
    return
  end if

  ! loop over the possible number of processors in the 1st direction
  loop_over_dim_procs : do procs_i = 2, max_procs(1)

    good = .false.
    
    ! check if this number of processors divides the total
    good = mod(total_procs,procs_i) == 0

    ! check that there would be the same number of points per processor in
    ! that direction
    good = good .and. ( mod(pp(itest(1))%points,procs_i) == 0)

    this_one_is_good : if (good) then

      ! this value works, so either stop if we are at the last level or call
      ! the routine again to check the next dimension
      procs_remain = total_procs / procs_i

      ! check that the number of points per proc would not be greater than the
      ! maximum number of points in a parallel scheme (e.g. 2 for the 
      ! fourth-order scheme with vpar).
      if ((pp(itest(1))%points / procs_i) <= pp(itest(1))%max_points) &
        & then

        ! terminate the recursion if we get to the end and pass back found or
        ! found_next together with the number of procs
        if (n == 1 .and. procs_remain == 1) then
           nprocs(1) = procs_i
           found = .true.
           return
        end if

        ! check the next direction
        if (n > 1) then
          do j=1, n-1
            itest_next(j) = itest(j+1)
            nprocs_next(j)= nprocs(j+1)
          end do
          call get_parallel_combination(&
              & procs_remain, n-1, itest_next, nprocs_next, found_next)
        end if

        ! If the next direction works, then all directions must work so unpack
        ! the number of processors for that direction into nprocs. The next
        ! level up will do the same again until it returns to the original
        ! calling routine.
        if (found_next) then
          nprocs(1) = procs_i
          do j=1, n-1
            nprocs(j+1) = nprocs_next(j)
          end do
          found = .true.
          return
        end if

      end if
      
    end if this_one_is_good

  end do loop_over_dim_procs

  ! We have already looped over all possibilities for the first direction,
  ! so there is no match.
  found = .false.
  return

end subroutine get_parallel_combination

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine fills in the parallel plan based on the number of
!> processors in each direction in iplan
!----------------------------------------------------------------------------

subroutine set_parallel_plan(itest, procs, n_parallel)

  !> array holding the indicies of the things to parallelize over
  integer, dimension(nplan), intent(in) :: itest
  !> the number of procs for the parallel directions
  integer, dimension(nplan), intent(in) :: procs
  !> the number of elements in itest over which we shall parallelize
  integer, intent(in) :: n_parallel

  integer :: i, j
  logical :: fill

  ! for all parallel directions, set the grid sizes and number of processors
  do i=1, n_parallel
    pp(itest(i))%parallel = .true.
    pp(itest(i))%procs    = procs(i)
    pp(itest(i))%points   = pp(itest(i))%points / pp(itest(i))%procs
  end do

  ! for all other directions, set the number of processors to 1
  do i=1, nplan
    fill = .true.
    do j=1, n_parallel
      if (i .eq. itest(j)) fill = .false.
    enddo
    if (fill) then
      pp(i)%parallel    = .false. 
      pp(i)%procs       = 1
    end if
  end do

end subroutine set_parallel_plan  

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine checks that the parallel layout is consistent for the
!> variables provided to the rest of the code
!----------------------------------------------------------------------------

subroutine check_parallel_layout
  
  use general, only : gkw_abort 
  use mpiinterface
  use control, only : vp_trap, collisions, zonal_adiabatic,non_linear,&
      & ltrapping_arakawa

  ! -- not implemented or tested--
  if (parallel_vpar) then
    if (vp_trap == 1) then
      call gkw_abort('check_parallel_layout:'//                              &
          & 'n_procs_vpar > 1 not tested with vp_trap=1?')
    end if
  end if !parallel v_par

  if(parallel_s) then
     call gkw_warn('parallel.dat must be reconstructed for parallel_s & 
         & with post processing script gkw_construct-parallel-output.sh.')

     if (vp_trap == 1) call gkw_abort('vp_trap not implemented for parallel_s')
  end if !parallel_s

  ! Arakawa type differencing not implemented for both parallel_s and paralllel_vpar 
  if (ltrapping_arakawa) then
    if (parallel_s .and. parallel_vpar) call gkw_abort('ltrapping_arakawa not'//&
        & 'implemented with both parallel_s and parallel_vpar')
  end if

  ! -- consistency checks --
  ! total number of procs selected
  if(n_procs_mu*n_procs_s*n_procs_sp*n_procs_vpar /=                         &
                                   & number_of_processors) then
    write(*,*) 'mu',n_procs_mu
    write(*,*) 'sp',n_procs_sp
    write(*,*) ' s',n_procs_s
    write(*,*) 'vp',n_procs_vpar
    write(*,*) 'nn',number_of_processors
    call gkw_abort('check_parallel_layout:'//                                &
                  &' n_procs_mu*n_procs_s*n_procs_sp*n_procs_vpar /='//      &
                  &' number_of_procs !')
  end if
  
  ! too many procs in particular direction
  if (n_procs_mu > n_mu_grid) then
    call gkw_abort('check_parallel_layout: too many procs for mu_grid')
  end if
  if (n_procs_sp > number_of_species) then
    call gkw_abort('check_parallel_layout:'//                                &
        &          ' too many procs for number of species')
  end if
  if (n_procs_s > n_s_grid) then
    call gkw_abort('check_parallel_layout: too many procs in s')
  end if
  if (n_procs_vpar > n_vpar_grid) then
    call gkw_abort('check_parallel_layout: too many procs in vpar')
  end if

  ! -- efficiency checks --
  ! all procs should have the same amount of work so quit if that is not true
  if (mod(n_mu_grid, n_procs_mu) /= 0) then
    call gkw_abort('check_parallel_layout:'//                                &
                  &' n_mu_grid is not a multiple of n_procs_mu')
  end if
  if (mod(n_s_grid, n_procs_s) /= 0)  then
    call gkw_abort('check_parallel_layout:'//                                &
                  &' ns is not a multiple of n_procs_s')
  end if
  if (mod(number_of_species, n_procs_sp) /= 0) then
    call gkw_abort('check_parallel_layout:'//                                &
                  &' number_of_species is not a multiple of n_procs_sp')
  end if
  if (mod(n_vpar_grid, n_procs_vpar) /= 0) then
    call gkw_abort('check_parallel_layout:'//                                &
                  &' n_vpar_grid is not a multiple of n_procs_vpar')
  end if

end subroutine check_parallel_layout

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> print the parallel plan to stdout
!----------------------------------------------------------------------------

subroutine parallel_report(a)

  use mpiinterface
  use control, only : vp_trap
  use global,  only : lverbose
  
  type (plan), intent(in), dimension(4) :: a

  integer :: i, ierr
  
  if (root_processor) then
    write (*,*) ' --- parallel report --- '
    write (*,*) 
    write (*,*) 'given:'
    write (*,*) ' number of processors =',number_of_processors
    write (*,*) 
    write (*,*) ' number of species    =',number_of_species
    write (*,*) ' n_mu_grid            =',n_mu_grid
    write (*,*) ' n_vpar_grid          =',n_vpar_grid
    write (*,*) ' n_s_grid             =',n_s_grid
    write (*,*)
    write (*,*) 'The code will:'
    write (*,*)

    do i=1, 4
  
     if(a(i)%parallel) then
       write (*,*) 'parallelise over ',a(i)%name
       write (*,*) '   with'
       write (*,*) 'procs           =',a(i)%procs
       write (*,*) 'points per proc =',a(i)%points
       write (*,*)
     end if
  
    end do
  
    if (lverbose) then
      write (*,*) ' per processor values:'
      write (*,*) '   nsp  =',nsp
      write (*,*) '   nmu  =',nmu
      write (*,*) '  nvpar =',nvpar
      write (*,*) '    ns  =',ns
      write (*,*) '         '
      write (*,*) ' arrays for the first and last points on the root processor'
      write (*,*) ' isppb     =',isppb
      write (*,*) ' isppe     =',isppe
      write (*,*) ' imupb     =',imupb
      write (*,*) ' imupe     =',imupe
      write (*,*) ' ivparpb   =',ivparpb
      write (*,*) ' ivparpe   =',ivparpe
    end if
  
    if (vp_trap .eq. 1) then
      write (*,*) ' (using vp_trap)'     
    end if
    
    write (*,*)
    write (*,*) ' --- end of parallel report ---'
  
  end if

end subroutine parallel_report

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Set up additional communicators for various hyperslabs in the cartesian
!> topology. When using MPI, any additional communicator should be correctly
!> initialised, irrespective of if it will be used or not. 
!----------------------------------------------------------------------------

subroutine setup_communicators
 
  use mpiinterface
  use mpicomms
  
  logical, dimension (nplan) :: remain_dims

  ! Default values; these assume there is no parallelisation in a particular
  ! direction e.g. COMM_VPAR_NE should contain only the calling process by
  ! default, as all the vpar points are local (there are no other vpar
  ! points); COMM_VPAR_EQ should contain all processors as they all solve for
  ! the same part of the vpar grid. MPI_COMM_SELF and COMM_CART are suitable
  ! communicators for these examples; these are provided by either MPI or a
  ! dummy value in mpiinterface.
  COMM_ALL_EQ     = MPI_COMM_SELF
  COMM_VPAR_NE    = MPI_COMM_SELF
  COMM_VPAR_EQ    = COMM_CART
  COMM_S_NE       = MPI_COMM_SELF
  COMM_S_EQ       = COMM_CART
  COMM_MU_NE      = MPI_COMM_SELF
  COMM_MU_EQ      = COMM_CART
  COMM_SP_EQ      = COMM_CART
  COMM_SP_NE      = MPI_COMM_SELF
  COMM_SP_EQ_S_EQ = COMM_CART
  COMM_VPAR_NE_MU_NE = MPI_COMM_SELF
  
  ! Sub-communicators for processors parallel (NE) and perpendicular (EQ) in
  ! the processor grid.
#if defined(mpi)
  if (number_of_processors > 1) then
    ! vpar
    remain_dims(:) = .false.
    if (pp(ivpar)%parallel) remain_dims(pp(ivpar)%direction) = .true.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_VPAR_NE,ierr)
    remain_dims(:) = (.not. remain_dims(:))    
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_VPAR_EQ,ierr)  
 
    ! s
    remain_dims(:) = .false.
    if (pp(is)%parallel) remain_dims(pp(is)%direction) = .true.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_S_NE,ierr)
    remain_dims(:) = (.not. remain_dims(:))
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_S_EQ,ierr)
 
    ! mu
    remain_dims(:) = .false.
    if (pp(imu)%parallel) remain_dims(pp(imu)%direction) = .true.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_MU_NE,ierr)
    remain_dims(:) = (.not. remain_dims(:))
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_MU_EQ,ierr)
 
    ! species
    remain_dims(:) = .false.
    if (pp(isp)%parallel) remain_dims(pp(isp)%direction) = .true.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_SP_NE,ierr)
    remain_dims(:) = (.not. remain_dims(:))
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_SP_EQ,ierr)
 
    ! mu and vpar
    remain_dims(:) = .false.
    if (pp(ivpar)%parallel) remain_dims(pp(ivpar)%direction) = .true.
    if (pp(imu)%parallel)   remain_dims(pp(imu)%direction)   = .true.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_VPAR_NE_MU_NE,ierr)
    
    ! equal s and species
    remain_dims(:) = .true.
    if (pp(isp)%parallel) remain_dims(pp(isp)%direction) = .false.
    if (pp(is)%parallel)  remain_dims(pp(is)%direction)  = .false.
    call MPI_CART_SUB(COMM_CART,remain_dims(1:n_dims),COMM_SP_EQ_S_EQ,ierr)
  end if
#endif

end subroutine setup_communicators

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Provide the ranks of neighbouring processors for directions in which
!> send/recv communications are required. This requires the additional
!> communicators of mpicomms to be initialised.
!----------------------------------------------------------------------------

subroutine get_local_ranks

  use mpiinterface
  use mpicomms
 
  ! Default values; MPI_PROC_NULL - a dummy process
  proc_vpar_prev          = MPI_PROC_NULL
  proc_vpar_next          = MPI_PROC_NULL
  proc_s_prev             = MPI_PROC_NULL
  proc_s_next             = MPI_PROC_NULL
  proc_mu_prev            = MPI_PROC_NULL
  proc_mu_next            = MPI_PROC_NULL
  proc_vpar_prev_mu_prev  = MPI_PROC_NULL
  proc_vpar_prev_mu_next  = MPI_PROC_NULL
  proc_vpar_next_mu_prev  = MPI_PROC_NULL
  proc_vpar_next_mu_next  = MPI_PROC_NULL
  
#ifdef mpi
  ! next and previous processor for mu, s and vpar
  if (pp(imu)%parallel) then
    call MPI_CART_SHIFT(COMM_MU_NE,0,1,proc_mu_prev,proc_mu_next,ierr)
  end if
  if (pp(is)%parallel)  then
    call MPI_CART_SHIFT(COMM_S_NE,0,1,proc_s_prev,proc_s_next,ierr)
  end if
  if (pp(ivpar)%parallel) then
    call MPI_CART_SHIFT(COMM_VPAR_NE,0,1,proc_vpar_prev,proc_vpar_next,ierr)
  end if
#endif

  ! The processors along diagonals in the grid can always be found by
  ! systematically going through the processors till the one responsible for
  ! the right part of the grid is found. However, we do something a little
  ! more complicated here...

  if (number_of_processors > 1) then
    proc_vpar_next_mu_next = cart_rank(dim1='vpar',s1=+1,dim2='mu',s2=+1)
    proc_vpar_next_mu_prev = cart_rank(dim1='vpar',s1=+1,dim2='mu',s2=-1)
    proc_vpar_prev_mu_next = cart_rank(dim1='vpar',s1=-1,dim2='mu',s2=+1)
    proc_vpar_prev_mu_prev = cart_rank(dim1='vpar',s1=-1,dim2='mu',s2=-1)
  end if
  
end subroutine get_local_ranks

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> find the rank in COMM_CART of the proc shifted in different directions
!----------------------------------------------------------------------------
  
function cart_rank(dim1,s1,dim2,s2,dim3,s3,dim4,s4)

  use mpiinterface
  use mpicomms

  character (len=*), intent(in) :: dim1,dim2,dim3,dim4
  integer, intent(in) :: s1,s2,s3,s4 
  optional :: dim2,s2,dim3,s3,dim4,s4
  integer :: cart_rank
  
  integer, dimension(nplan) :: coords_in
  logical :: change_null
  integer :: j
  
  coords_in(:) = coords(:)
  j=1
  change_null = .true.
  check_shift : do i=1,nplan
    if (pp(i)%parallel) then
        if (pp(i)%name==dim1) then
          coords_in(j)=coords_in(j)+s1
          if (coords_in(j) < 0 .or. coords_in(j) >= pp(i)%procs) then
            change_null = .false.
            exit check_shift
          end if
        end if
      if (present(dim2) .and. present(s2)) then
        if (pp(i)%name==dim2) then
          coords_in(j)=coords_in(j)+s2
          if (coords_in(j) < 0 .or. coords_in(j) >= pp(i)%procs) then
            change_null = .false.
            exit check_shift
          end if
        end if
      end if
      if (present(dim3) .and. present(s3)) then
        if (pp(i)%name==dim3) then
          coords_in(j)=coords_in(j)+s3
          if (coords_in(j) < 0 .or. coords_in(j) >= pp(i)%procs) then
            change_null = .false.
            exit check_shift
          end if
        end if
      end if
      if (present(dim4) .and. present(s4)) then
        if (pp(i)%name==dim4) then
          coords_in(j)=coords_in(j)+s4
          if (coords_in(j) < 0 .or. coords_in(j) >= pp(i)%procs) then
            change_null = .false.
            exit check_shift
          end if
        end if
      end if
      j=j+1
    end if
  enddo check_shift

  ! default
  cart_rank = MPI_PROC_NULL
  
  ! This avoids the error of attempting to obtain the rank if the coords are
  ! not in the processor grid.
#ifdef mpi
  if (change_null) call MPI_CART_RANK(COMM_CART,coords_in,cart_rank,j)
#endif

end function cart_rank

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Create a cartesian processor topology (communicator COMM_CART) and get the
!> coordinates of the local processor within the processor grid (coords).
!----------------------------------------------------------------------------

subroutine setup_proc_coords

  use mpiinterface
  use mpicomms
  use general, only : gkw_abort 

  integer :: i,j,parallel_dim_count
  integer, parameter :: impi_dummy =-i_huge+432523
  integer :: impi_dummy_comm
  
#if defined(mpi)
  impi_dummy_comm = MPI_COMM_WORLD
#else
  impi_dummy_comm = impi_dummy
#endif

  ! state which directions should be periodic or not
  pp(:)%periodic      =   .false.
  pp(is)%periodic     =   .true.

  
  ! find out how many dimensions the cartesian domain has
  n_dims = count(pp(:)%parallel)

  ! consistency check counter
  parallel_dim_count = 0

  do i=1, nplan

    if (pp(i)%parallel) then
      ! MPI direction indexing starts at 1
      parallel_dim_count = parallel_dim_count + 1

      pp(i)%direction = parallel_dim_count
      dims(parallel_dim_count)     = pp(i)%procs
      periodic(parallel_dim_count) = pp(i)%periodic

    end if
    
  end do
  
  if (parallel_dim_count /= n_dims) then
    call gkw_abort('setup_proc_coords: bad parallel_dim_count')
  end if

  coords = 0
  COMM_CART = impi_dummy_comm
  
#if defined(mpi)
  ! create the topology; give the communicator a value in all cases
  if (number_of_processors > 1) then
    call MPI_CART_CREATE(MPI_COMM_WORLD,n_dims,dims(1:n_dims),               &
        &                periodic(1:n_dims),reorder,COMM_CART,ierr)
    
    ! get the coordinates of the local processor in this new grid
    call MPI_CART_COORDS(COMM_CART,processor_number,nplan,coords,ierr)
  end if
#endif

  ! copy the coords into the plan structure
  j=1
  do i=1,nplan
    if (pp(i)%parallel) then
      pp(i)%iproc=coords(j)
      j=j+1
    else
      pp(i)%iproc=0
    end if
  enddo

  ! provide the s position in the processor grid (APS: is this required?)

  iproc_s = pp(is)%iproc

end subroutine setup_proc_coords

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Create integers corresponding to the first and last points in each
!> direction that the local processor is responsible for. Provide logicals
!> indicating if a processor is responsible for a upper/lowel boundary in the
!> computation domain.
!----------------------------------------------------------------------------

subroutine setup_local_grid_ranges

  use control, only : vp_trap

  integer :: i,j

  !
  ! Use the coords array and the number of points in each direction to
  ! determine the start and end points.
  !

  j=1
  do i=1,nplan
    if (pp(i)%parallel) then
      if (i == ivpar .and. vp_trap == 1) then
        pp(i)%ipb=coords(j)*(pp(i)%points/2) + 1
        pp(i)%ipe=(coords(j)+1)*(pp(i)%points/2)
      else
        pp(i)%ipb=coords(j)*pp(i)%points + 1
        pp(i)%ipe=(coords(j)+1)*pp(i)%points
      end if
      j=j+1
    else
      pp(i)%ipb=1
      pp(i)%ipe=pp(i)%points
    end if
  enddo

  !
  ! copy the values into the variables used elsewhere in the code
  !
  
  isppb   = pp(isp)%ipb
  isppe   = pp(isp)%ipe
  imupb   = pp(imu)%ipb
  imupe   = pp(imu)%ipe
  ivparpb = pp(ivpar)%ipb
  ivparpe = pp(ivpar)%ipe
  ispb    = pp(is)%ipb
  ispe    = pp(is)%ipe
  ixpb    = 1  
  !
  ! calculate logicals for convenience when dealing with boundaries
  !

  lproc_s_lowerb    = (ispb    == 1)
  lproc_s_upperb    = (ispe    == n_s_grid)
  if (vp_trap == 1) then
    lproc_vpar_lowerb = (ivparpe == n_vpar_grid)
    lproc_vpar_upperb = (ivparpe == n_vpar_grid)
  else
    lproc_vpar_lowerb = (ivparpb == 1)
    lproc_vpar_upperb = (ivparpe == n_vpar_grid)
  end if
  lproc_mu_lowerb   = (imupb   == 1)
  lproc_mu_upperb   = (imupe   == n_mu_grid)
  
end subroutine setup_local_grid_ranges

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> It appears that sometimes when performing MPI_CART_SHIFT on a sub cartesian
!> topology, openmpi forgets which dimensions should be periodic. This routine
!> does not fix the topology; it just identifies the processors required for
!> communication in periodic directions. This is done by communicating the
!> ranks of the first and last processors in the grid to eachother.
!----------------------------------------------------------------------------

subroutine openmpi_fix

  use mpicomms
  use mpiinterface

  integer :: rank, flag, sumflag
  
  ! s-direction
  if (pp(is)%periodic .and. pp(is)%parallel) then
    
    ! get my rank
    call mpicomm_rank(COMM_S_NE,rank)
    
    ! I am at the start of the grid; the processor at the end obtains my rank.
    flag = -1
    if (pp(is)%ipb == 1) flag = rank
    call mpiallreduce_max(flag,sumflag,1,COMM_S_NE)
    if (pp(is)%ipe == n_s_grid) proc_s_next = sumflag

    ! I am at the end of the grid; the processor at the start obtains  my rank.
    flag = -1
    if (pp(is)%ipe == n_s_grid) flag = rank
    call mpiallreduce_max(flag,sumflag,1,COMM_S_NE)
    if (pp(is)%ipb == 1) proc_s_prev = sumflag

  end if

end subroutine openmpi_fix

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module grid
