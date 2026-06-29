!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Provide an interface to the mpi parameters and functions.
!> !$Id: mpiinterface.F90 1023 2009-07-02 20:27:02Z phsgbq $
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
module mpiinterface
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
! 2008-01-25  : coded
!
  
  use global

  implicit none

  private :: mpiwtick, mpiwtime, get_basic_mpi_params, ierr
  
  interface mpibarrier
    module procedure mpibarrier_comm
  endinterface

  interface mpibcast_logical
    module procedure mpibcast_logical_scalar
  endinterface

  interface mpibcast_real
    module procedure mpibcast_real_scalar
    module procedure mpibcast_real_array1
    module procedure mpibcast_real_array2
    module procedure mpibcast_real_array3
  endinterface

  interface mpibcast_integer
    module procedure mpibcast_integer_scalar
  endinterface
    
  interface mpibcast_character
    module procedure mpibcast_character_scalar
  endinterface

  interface mpiallreduce_sum
    module procedure mpiallreduce_sum_r_array_1
    module procedure mpiallreduce_sum_c_scalar
    module procedure mpiallreduce_sum_c_array_1
  endinterface

  interface mpiallreduce_max
    module procedure mpiallreduce_max_int_scalar
    module procedure mpiallreduce_max_real_scalar
  endinterface

  interface mpiallreduce_min
    module procedure mpiallreduce_min_int_scalar
    module procedure mpiallreduce_min_r_scalar
  endinterface

  interface mpiallreduce_or
    module procedure mpiallreduce_or_logical_scalar
  endinterface

  interface mpicart_coords_self
    module procedure mpicart_coords_self_1d
  endinterface

  interface gather_array
    module procedure gather_1d_real
    module procedure gather_2d_real
  end interface

! Swap variables and routines depending on if we use mpi or not.

#if defined(mpi)
  include 'mpif.h'

  ! lines below stops us accidently using these in the code; eventually
  ! make everything in this module private by default
  private :: MPI_DOUBLE_PRECISION, MPI_REAL, MPI_COMPLEX, parallel_run

#else

  ! alternatives to the mpi routines
  interface MPI_WTIME
    module procedure mpiwtime
  endinterface
  interface MPI_WTICK
    module procedure mpiwtick
  endinterface
  !> dummy MPI_STATUS_SIZE
  integer, parameter :: MPI_STATUS_SIZE = 1
  !> dummy MPI_PROC_NULL
  integer, parameter :: MPI_PROC_NULL = -1
  !> default offset kind
  integer, parameter :: MPI_OFFSET_KIND = KIND(1.0D0)

#endif

  !> total number of processors
  integer :: number_of_processors
  !> local processor rank
  integer :: processor_number 
  !> for mpi error
  integer :: ierr
  !> true for a parallel run
  logical :: parallel_run
  !> root process
  logical :: root_processor
  !> the process with the largest rank
  logical :: last_processor
  !> Is MPI ready to use?
  logical :: lmpi_is_ready = .false.

! set MPIREAL_X etc. to some MPI precision determined at compile time

#if defined(mpi)


#if defined(real_precision_8)
  !> the MPI type corresponding to the default real type used
  integer, parameter :: MPIREAL_X = MPI_REAL8
  integer, parameter :: MPI2REAL_X = MPI_2DOUBLE_PRECISION
  !> the MPI type corresponding to the default complex type used
  integer, parameter :: MPICOMPLEX_X = MPI_COMPLEX16
#endif     

#if defined(real_precision_4)
  integer, parameter :: MPIREAL_X = MPI_REAL4
  integer, parameter :: MPI2REAL_X = MPI_2REAL
  integer, parameter :: MPICOMPLEX_X = MPI_COMPLEX8
#endif

#if defined(real_precision_double)
  integer, parameter :: MPIREAL_X = MPI_DOUBLE_PRECISION
  integer, parameter :: MPI2REAL_X = MPI_2DOUBLE_PRECISION
  integer, parameter :: MPICOMPLEX_X = MPI_DOUBLE_COMPLEX
#endif

#if defined(real_precision_default)
  integer, parameter :: MPIREAL_X = MPI_REAL
  integer, parameter :: MPI2REAL_X = MPI_2REAL
  integer, parameter :: MPICOMPLEX_X = MPI_COMPLEX
#endif

#else
  integer, parameter :: MPICOMPLEX_X  = I_RUN_WITHOUT_MPI
  integer, parameter :: MPIREAL_X     = I_RUN_WITHOUT_MPI 
  integer, parameter :: MPI2REAL_X    = I_RUN_WITHOUT_MPI
  integer, parameter :: MPI_COMM_SELF = I_RUN_WITHOUT_MPI 
#endif

contains

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> The basic mpi parameters are obtained here. This provides the number of
!> processors and the processor number to the rest of the code.
!-----------------------------------------------------------------------------

  subroutine get_basic_mpi_params()

#if defined(mpi)

    CALL MPI_COMM_SIZE(MPI_COMM_WORLD, number_of_processors, ierr)
    CALL MPI_COMM_RANK(MPI_COMM_WORLD, processor_number, ierr)

#else

    number_of_processors = 1
    processor_number = 0

#endif 

    parallel_run = (number_of_processors > 1)

  end subroutine get_basic_mpi_params

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> find the coordinate in one direction
!-----------------------------------------------------------------------------

subroutine mpicart_coords_self_1d(ICOMM,coord)

  integer, intent(in) :: ICOMM
  integer, intent(out) :: coord

  integer :: rank,comm
  integer, parameter :: maxdims = 1

#ifdef mpi
  comm=ICOMM
  if (number_of_processors > 1) then
    call mpicomm_rank(comm,rank)
    call MPI_CART_COORDS(comm,rank,maxdims,coord,ierr)
  else
    coord = 0
  end if
#else
  coord = 0
#endif

end subroutine mpicart_coords_self_1d

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> MPI_WTIME substitute if no mpi implementation is available
!-----------------------------------------------------------------------------

  function mpiwtime()

    double precision :: mpiwtime
    integer :: ccount, ccount_rate
  
    call system_clock(ccount, ccount_rate)
    
    if (ccount_rate == 0) then
      mpiwtime = 0.
    else 
      mpiwtime = (1.*ccount) / ccount_rate
    endif
  
  end function mpiwtime

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> MPI_WTICK substitute if no mpi implementation is available
!-----------------------------------------------------------------------------

  function mpiwtick()
  
    double precision :: mpiwtick
    integer :: ccount_rate
  
    call system_clock(COUNT_RATE=ccount_rate)
    
    if (ccount_rate == 0) then
      mpiwtick = 0.
    else
      mpiwtick = 1./ccount_rate
    endif
  
  end function mpiwtick

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match LOGICAL SCALAR for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_logical_scalar(lscalar,nscalar,IPROC)

  logical, intent(inout) :: lscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(lscalar,nscalar,MPI_LOGICAL,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_logical_scalar

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match INTEGER SCALAR for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_integer_scalar(iscalar,nscalar,IPROC)

  integer, intent(inout) :: iscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(iscalar,nscalar,MPI_INTEGER,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_integer_scalar

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match REAL SCALAR for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_real_scalar(rscalar,nscalar,IPROC)

  real, intent(inout) :: rscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(rscalar,nscalar,MPIREAL_X,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_real_scalar

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match REAL ARRAY for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_real_array1(rscalar,nscalar,IPROC)

  real, dimension(:), intent(inout) :: rscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(rscalar,nscalar,MPIREAL_X,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_real_array1

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match REAL ARRAY for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_real_array2(rscalar,nscalar,IPROC)

  real, dimension(:,:), intent(inout) :: rscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(rscalar,nscalar,MPIREAL_X,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_real_array2

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match REAL ARRAY for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_real_array3(rscalar,nscalar,IPROC)

  real, dimension(:,:,:), intent(inout) :: rscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(rscalar,nscalar,MPIREAL_X,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_real_array3

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> match CHARACTER SCALAR for MPI_BCAST
!----------------------------------------------------------------------------

subroutine mpibcast_character_scalar(cscalar,nscalar,IPROC)

  character (len=*), intent(inout) :: cscalar
  integer, intent(in)    :: nscalar
  integer, optional,  intent(in) :: IPROC

#if defined(mpi)

  integer :: iproc_bcast
     
  iproc_bcast = 0
  if (present(IPROC)) iproc_bcast = IPROC
  call MPI_BCAST(cscalar,nscalar,MPI_CHARACTER,iproc_bcast,MPI_COMM_WORLD,ierr)

#endif

end subroutine mpibcast_character_scalar

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine mpifinalize(i_abort)

    integer, optional, intent(in) :: i_abort
  
#if defined(mpi)

    ! FIX CONTROL SO THAT WE DON'T NEED MPI_ABORT
    !if (present(i_abort)) call MPI_ABORT()
    call mpibarrier()
    call MPI_FINALIZE(ierr)

#endif

    lmpi_is_ready = .false.

  end subroutine mpifinalize

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine mpiinit()

#if defined(mpi)

    call MPI_INIT(ierr)

#endif

    ! set processor_number and number_of_processors
    call get_basic_mpi_params

    ! set the root_processor and last_processor logicals
    root_processor = ( processor_number == 0 )
    last_processor = ( processor_number == number_of_processors - 1 )  
    if (root_processor) then
      write(*,*)
      write(*,*)'Running on ',number_of_processors,' processors'
      write(*,*)
    endif

    lmpi_is_ready = .true.

  end subroutine mpiinit

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> call MPI_BARRIER with communicator icomm, or just MPI_COMM_WORLD
!-----------------------------------------------------------------------------

subroutine mpibarrier_comm(icomm)
  
  integer, optional, intent(in) :: icomm
  integer :: icom

#ifdef mpi
    icom = MPI_COMM_WORLD
    if (present(icomm)) icom = icomm
    call MPI_BARRIER(icom,ierr)
#endif

end subroutine mpibarrier_comm

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> call MPI_COMM_RANK
!-----------------------------------------------------------------------------

subroutine mpicomm_rank(icomm,rank)

  integer, intent(in)  :: icomm
  integer, intent(out) :: rank

#ifdef mpi
    call MPI_COMM_RANK(icomm,rank,ierr)
#else
    rank = 0
#endif

end subroutine mpicomm_rank


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for mpi_allreduce MAXLOC
!-----------------------------------------------------------------------------

subroutine mpiallreduce_maxloc(a,a_max,n,icomm_in)

  real, dimension(2), intent(in) :: a
  real, dimension(2), intent(out) :: a_max
  integer, intent(in) :: n
  integer, optional, intent(in) :: icomm_in
  integer :: icomm
  
#ifdef mpi
  icomm = MPI_COMM_WORLD
  if (present(icomm_in)) icomm = icomm_in
  call MPI_ALLREDUCE(a,a_max,1,MPI2REAL_X,MPI_MAXLOC,icomm,ierr)
#else
  a_max = a
#endif
  

end subroutine mpiallreduce_maxloc

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for logical scalar .or. MPI_ALLREDUCE
!-----------------------------------------------------------------------------

subroutine mpiallreduce_or_logical_scalar(a,or_a,n_a,icomm_in)

  logical, intent(in) :: a
  integer, intent(in) :: n_a
  logical, intent(out) :: or_a
  integer, optional, intent(in) :: icomm_in
  integer :: icomm

#ifdef mpi
  icomm = MPI_COMM_WORLD
  if (present(icomm_in)) icomm = icomm_in
  call MPI_ALLREDUCE(a,or_a,1,MPI_LOGICAL,MPI_LOR,icomm,ierr)
#else
  or_a = a
#endif

end subroutine mpiallreduce_or_logical_scalar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for integer scalar max MPI_ALLREDUCE
!-----------------------------------------------------------------------------
  
subroutine mpiallreduce_max_int_scalar(a,max_a,n_a,icomm_in)
 
  integer, intent(in) :: a,n_a
  integer, intent(out) :: max_a
  integer, optional, intent(in) :: icomm_in
  integer :: icomm

#ifdef mpi
  icomm = MPI_COMM_WORLD
  if (present(icomm_in)) icomm = icomm_in
  call MPI_ALLREDUCE(a,max_a,1,MPI_INTEGER,MPI_MAX,icomm,ierr)
#else
  max_a = a
#endif

end subroutine mpiallreduce_max_int_scalar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for real scalar MPI_ALLREDUCE
!-----------------------------------------------------------------------------
  
subroutine mpiallreduce_max_real_scalar(a,max_a,n_a,icomm_in)
 
  real, intent(in) :: a
  integer, intent(in) :: n_a
  real, intent(out) :: max_a
  integer, optional, intent(in) :: icomm_in
  integer :: icomm

#ifdef mpi
  icomm=MPI_COMM_WORLD
  if (present(icomm_in)) icomm=icomm_in
  call MPI_ALLREDUCE(a,max_a,1,MPI_REAL,MPI_MAX,icomm,ierr)
#else
  max_a=a
#endif

end subroutine mpiallreduce_max_real_scalar
  
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for integer scalar MPI_ALLREDUCE
!-----------------------------------------------------------------------------
  
subroutine mpiallreduce_min_int_scalar(a,max_a,n_a,icomm_in)
 
  integer, intent(in) :: a, n_a
  integer, intent(out) :: max_a
  integer, optional, intent(in) :: icomm_in
  integer :: icomm
  
#ifdef mpi
  icomm=MPI_COMM_WORLD
  if (present(icomm_in)) icomm=icomm_in
  call MPI_ALLREDUCE(a,max_a,1,MPI_INTEGER,MPI_MIN,icomm,ierr)
#else
  max_a=a
#endif

end subroutine mpiallreduce_min_int_scalar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> wrapper for real scalar MPI_ALLREDUCE
!-----------------------------------------------------------------------------
   
subroutine mpiallreduce_min_r_scalar(a,min_a,n_a,icomm_in)
 
  real, intent(in) :: a
  integer, intent(in) :: n_a
  real, intent(out) :: min_a
  integer, optional, intent(in) :: icomm_in
  integer :: icomm
  
#ifdef mpi
  icomm=MPI_COMM_WORLD
  if (present(icomm_in)) icomm=icomm_in
  call MPI_ALLREDUCE(a,min_a,1,MPIREAL_X,MPI_MIN,icomm,ierr)
#else
  min_a=a
#endif

end subroutine mpiallreduce_min_r_scalar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
subroutine mpiallreduce_sum_c_scalar(a,sum_a,n_elements,icomm)

  integer, intent(in) :: n_elements
  complex, intent(in)  :: a
  complex, intent(out) :: sum_a
  integer, optional, intent(in) :: icomm
  integer :: i_comm

  if (parallel_run) then
#ifdef mpi
    i_comm=MPI_COMM_WORLD
    if (present(icomm)) i_comm=icomm
    call MPI_ALLREDUCE(a,sum_a,1,MPICOMPLEX_X,MPI_SUM,i_comm,ierr)
#endif
  else
    sum_a=a
  end if
  
end subroutine mpiallreduce_sum_c_scalar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
subroutine mpiallreduce_sum_c_array_1(a,sum_a,n_elements,icomm)

  integer, intent(in) :: n_elements
  complex, dimension(:), intent(in)  :: a
  complex, dimension(:), intent(out) :: sum_a
  integer, optional, intent(in) :: icomm
  integer :: i_comm

  if (parallel_run) then
#ifdef mpi
    i_comm=MPI_COMM_WORLD
    if (present(icomm)) i_comm=icomm
    call MPI_ALLREDUCE(a,sum_a,n_elements,MPICOMPLEX_X,MPI_SUM,i_comm,ierr)
#endif
  else
    sum_a=a
  end if
  
end subroutine mpiallreduce_sum_c_array_1

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine mpiallreduce_sum_r_array_1(a,sum_a,n_elements,icomm)

  integer, intent(in) :: n_elements
  real, dimension(n_elements), intent(in)  :: a
  real, dimension(n_elements), intent(out) :: sum_a
  integer, optional, intent(in) :: icomm
  integer :: i_comm

  if (parallel_run) then
#ifdef mpi
    i_comm=MPI_COMM_WORLD
    if (present(icomm)) i_comm=icomm
    call MPI_ALLREDUCE(a,sum_a,n_elements,MPIREAL_X,MPI_SUM,i_comm,ierr)
#endif
  else
    sum_a=a
  end if
  
end subroutine mpiallreduce_sum_r_array_1
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine mpiallreduce_sum_cmplx_array_1(a,sum_a,n_elements,icomm)
  
    integer, intent(in) :: n_elements
    complex, dimension(n_elements), intent(in)  :: a
    complex, dimension(n_elements), intent(out) :: sum_a
    integer, optional, intent(in) :: icomm
  
    integer :: i_comm
  
    if (parallel_run) then
    
#if defined(mpi)
      if (present(icomm)) then
        i_comm = icomm
      else
        i_comm = MPI_COMM_WORLD
      endif
      call MPI_ALLREDUCE(a, sum_a, n_elements, MPICOMPLEX_X, MPI_SUM,       &
          &              i_comm, ierr)
#endif

    else
    
      sum_a = a
  
    endif
  
  end subroutine mpiallreduce_sum_cmplx_array_1

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Get the displacement _idisp_ from the beginning of the file associated
!> with _ilun_ of the file pointer on processor _iproc_, for the communicator
!> _icomm_. This is useful for setting the next file view if only 1 processor
!> has written data to a file open for parallel write.
!----------------------------------------------------------------------------

  subroutine mpigetfiledisp(ilun,iproc,idisp,icomm)
  
    integer, intent(in) :: iproc, ilun
    
#if defined(mpi)

    integer (KIND=MPI_OFFSET_KIND), intent(out) :: idisp
    integer (KIND=MPI_OFFSET_KIND) :: offset
    integer, optional, intent(in) :: icomm
    integer :: i_comm

    i_comm = MPI_COMM_WORLD
    if (present(icomm)) i_comm = icomm
    
    if (processor_number == iproc) then
       call MPI_FILE_GET_POSITION(ilun,offset,ierr)
       call MPI_FILE_GET_BYTE_OFFSET(ilun,offset,idisp,ierr)
    end if
  
    ! 8 integers should be long enough?
    call MPI_BCAST(idisp,8,MPI_INTEGER,iproc,i_comm,ierr)
    
#else

    integer, intent(in) :: idisp
    integer, optional, intent(in) :: icomm
    ! do nothing

#endif

  end subroutine mpigetfiledisp

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!****************************************************************************
!> Gather 1d real array into a global array. This routine should only be
!> called by processes responsible for the different parts of the array.
!> The local array, local and global array sizes, communicators are input.
!> A global array is returned (on the processor with rank = 0 in the comm).
!----------------------------------------------------------------------------

subroutine gather_1d_real(global_x,n_x_grid,local_x,nx,COMM_X,ALLGATHER)

  integer, intent(in) :: n_x_grid,nx,COMM_X
  real, dimension(n_x_grid), intent(out) :: global_x
  real, dimension(nx), intent(in) :: local_x
  logical, optional :: ALLGATHER
  
  ! gather in the X-direction
  if (n_x_grid > nx) then
#ifdef mpi
  if (present(ALLGATHER)) then
    if (ALLGATHER) then
      call MPI_ALLGATHER(local_x,nx,MPIREAL_X,global_x,nx,MPIREAL_X,COMM_X,ierr)
    else
      call MPI_GATHER(local_x,nx,MPIREAL_X,global_x,nx,MPIREAL_X,0,COMM_X,ierr)
    end if
  else
    call MPI_GATHER(local_x,nx,MPIREAL_X,global_x,nx,MPIREAL_X,0,COMM_X,ierr)
  end if
#endif
  else
    global_x(:) = local_x(:)
  end if

end subroutine gather_1d_real

!****************************************************************************
!> Gather 2d real slices in a global slice. This routine should (usually)
!> only be called by processes responsible for the slice. The local array,
!> local and global array sizes, communicators are input. A
!> global array is returned (on the processor with rank = 0 in both comms).
!----------------------------------------------------------------------------

subroutine gather_2d_real(global_x_y,n_x_grid,n_y_grid,local_x_y,nx,ny,      &
    &                           COMM_X,COMM_Y)

  integer, intent(in) :: n_x_grid,n_y_grid,nx,ny,COMM_X,COMM_Y
  real, dimension(n_x_grid,n_y_grid), intent(out) :: global_x_y
  real, dimension(nx,ny), intent(in) :: local_x_y
  real, dimension(n_x_grid*n_y_grid) :: global_x_y_tmp
  real, dimension(nx,n_y_grid) :: part_x_y  
  integer :: ind,i,j,k,iproc
  
  ! first gather in the Y-direction
  if (n_y_grid > ny) then
#ifdef mpi
    call MPI_GATHER(local_x_y,nx*ny,MPIREAL_X,part_x_y,nx*ny,MPIREAL_X,0,    &
        &           COMM_Y,ierr)
#endif
  else
    part_x_y(:,:) = local_x_y(:,:)
  end if
      
  ! gather in the X-direction
  if (n_x_grid > nx) then
#ifdef mpi
    call MPI_GATHER(part_x_y,nx*n_y_grid,MPIREAL_X,global_x_y_tmp,           &
        & nx*n_y_grid,MPIREAL_X,0,COMM_X,ierr)
#else
    ind = 0
    do j=1,n_y_grid
      do i=1,nx
        ind=ind+1
        global_x_y_tmp(ind) = part_x_y(i,j)
      end do
    end do
#endif

    ! Need to re-order the gathered data from the second gather; the
    ! first gather is in the correct order.
    ind=0
    do k=1,(n_x_grid/nx)
      do j=1,n_y_grid
        do i=1+(k-1)*nx,k*nx
          ind=ind+1
          global_x_y(i,j) = global_x_y_tmp(ind)
        end do
      end do
    end do
  else
    global_x_y(:,:) = part_x_y(:,:)
  end if

end subroutine gather_2d_real

!****************************************************************************

end module mpiinterface
