!$Id: general.F90 1005 2009-07-02 16:12:03Z  $
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> The most general purpose routines, having no dependencies other than things
!> in mpiinterface, are (or should be) in here. Perhaps we could put
!> lverbose into a block.
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
module general
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  use global
  use mpiinterface
  
  implicit none

  private

  public :: svn_id
  public :: general_init
  public :: gkw_abort, gkw_warn, gkw_clean_abort
  public :: genfilename
  public :: matout
  public :: time_est
  public :: int2char

  interface gkw_abort
    module procedure gkw_abort_safe
    module procedure gkw_abort_check_iostat
  end interface
  
  interface genfilename
    module procedure genfilename1
    module procedure genfilename2
    module procedure create_filename
 end interface

  contains

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> initialize anything for general
!-----------------------------------------------------------------------------
subroutine general_init()
!-----------------------------------------------------------------------------

  call svn_id('$Id: general.F90 1005 2009-07-02 16:12:03Z  $')

end subroutine general_init

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Report the svn id (could be done by accumulation, then reported all at
!> once).
!-----------------------------------------------------------------------------
subroutine svn_id(id)
!-----------------------------------------------------------------------------

  character (len=*), intent(in) :: id

end subroutine svn_id

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Abort the code with an optional error message assuming all processors call
!> the routine.
!----------------------------------------------------------------------------

subroutine gkw_abort_safe(abort_message) 

  character (len=*),intent(in), optional :: abort_message

  if (present(abort_message)) then
    write (*,'(A,A,A,I4)') ' GKW_ABORT -- ', abort_message,&
                         & '  ', processor_number
  else
    write (*,'(A,A,A,I4)') ' GKW_ABORT -- ', '(no message given!)',&
                         & '  ', processor_number
  end if

  call mpifinalize(1)
  stop  

end subroutine gkw_abort_safe

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Abort the code; should only be used when all processors will call
!----------------------------------------------------------------------------

subroutine gkw_clean_abort(abort_message) 

  character (len=*),intent(in), optional :: abort_message

  if (root_processor) then
    if (present(abort_message)) then
      write (*,'(A,A,A)') ' GKW_ABORT -- ', abort_message,&
                           & '  '
    else
      write (*,'(A,A,A)') ' GKW_ABORT -- ', '(no message given!)',&
                           & '  '
    end if
  end if
  
  call mpifinalize(1)
  stop  

end subroutine gkw_clean_abort

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check if io_stat is non zero, if so then abort 
!----------------------------------------------------------------------------

subroutine gkw_abort_check_iostat(abort_message,io_stat)

  character (len=*),intent(in), optional :: abort_message
  integer, intent(in) :: io_stat

  integer :: io_stat_max,io_stat_min

  call mpiallreduce_max(io_stat,io_stat_max,1)
  call mpiallreduce_min(io_stat,io_stat_min,1)

  if (io_stat_min == 0 .and. io_stat_max == 0) return

  if (present(abort_message)) then
    if (root_processor) then
      write (*,'(A,A,A,I4)') ' GKW_ABORT -- ',abort_message,                 &
          &                  '; iostat=',io_stat

    endif
  else
    if (root_processor) then
      write (*,'(A,A,A,I4)') ' GKW_ABORT -- ', '(no message given!)',        &
          &                  '; iostat =',io_stat
    endif
  end if

  ! nice finalize
  call mpifinalize()
  stop

end subroutine gkw_abort_check_iostat

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine prints a warning message
!-----------------------------------------------------------------------------
subroutine gkw_warn(warning_message)  
!-----------------------------------------------------------------------------

  character (len=*),intent(in), optional :: warning_message 

  if (root_processor) then
    if ( present(warning_message) ) then
      write (*,'(A,A)') '*** WARNING *** ', warning_message
    else
      write (*,'(A)') '*** WARNING *** (no message given!)'
    end if
  end if

end subroutine gkw_warn

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> this routine returns a filename `filename', starting with the first 3
!> characters of `start', followed by a 6 digit representation of the input
!> integer `ival'. ival is inc. by 1 on return
!-----------------------------------------------------------------------------
subroutine genfilename1(start,filename,ival)
!-----------------------------------------------------------------------------

  character (len=3), intent(in) :: start
  integer, intent(inout) :: ival 
  character (len=9), intent(out) :: filename 

  write(filename,'(A,I6.6)')start,ival
  ival = ival + 1

end subroutine genfilename1 

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> this routine returns a filename `filename', starting with the first 5
!> characters of `start', followed by a 4 digit representation of each of the
!> two input integers `ival1' and `ival2', separated with `_'. 
!-----------------------------------------------------------------------------
subroutine genfilename2(start,filename,ival,ival2)
!-----------------------------------------------------------------------------

  character (len=5),  intent(in)  :: start
  integer, intent(in) :: ival,ival2 
  character (len=14), intent(out) :: filename 

  write(filename,'(A,I4.4,"_",I4.4)')start,ival,ival2
 
end subroutine genfilename2 

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> this routine returns a filename `filename', starting with the first 3
!> characters of `start', followed by a 6 digit representation of the input
!> integer `ival'. ival is inc. by 1 on return
!-----------------------------------------------------------------------------
subroutine create_filename(start,ival,length,filename)
!-----------------------------------------------------------------------------

  character (len=*), intent(in) :: start
  integer, intent(in) :: ival,length
  character (len=length), intent(out) :: filename

  filename(1:len(start))=start
  write(filename(len(start)+1:len(start)+8),'(I8.8)')ival

end subroutine create_filename

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine time_est(mat_elem,iii) 
!-----------------------------------------------------------------------------
!> This small routine keeps track of the matrix elements in order 
!> for the time step estimate 
!-----------------------------------------------------------------------------

  integer, intent(in) :: iii
  complex :: mat_elem 
  real, save :: store = 0.  !For individual terms 
  real, save :: store_all = 0. !For all terms


  select case(iii)

    case(0) !Reset the per term timestep estimate
      store = 0.E0
    case(1) !When called from put_element, update
      store = max(store, abs(mat_elem))
      store_all = max(store_all, abs(mat_elem))
    case(2) !Return the max value from the store
      mat_elem = store
    case(88) !Only update the global estimate
      store_all = max(store_all, abs(mat_elem))
    case(99) !return the maximum value from store_all
      !note only the real part is meaningful
      mat_elem = store_all
    case default 
      call gkw_abort('time_est: error in case iii') 
  
  end select 

end subroutine time_est

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine matout(iih,jjh,mat_elem) 
!------------------------------------------------------------------------------
! Simple output routine that is only used for testing the matix
! elements 
!------------------------------------------------------------------------------

  integer, intent(in) :: iih, jjh 
  complex, intent(in) :: mat_elem 
  
  integer, save :: init = 43423
  
  if (init == 0) then
    open(init, file = 'effe.dat') 
  endif 
  
  write(init,fmt='(I6,1X,I6,1X,2(1e13.5,1X))') iih, jjh, mat_elem
  
end subroutine matout 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function int2char(i)

  integer, intent(in) :: i
  character (len=8) :: int2char

  write(int2char,'(I8.8)') i

end function int2char

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module general
