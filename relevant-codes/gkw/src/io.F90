! $Id$
!> deals with some input and output related things
!============================================================================

module io

  use mpiinterface
  use global

  implicit none
  
  private

  public :: decide_who_writes, get_free_lun, open_file, close_output_files
  public :: close_file,file_is_open,flush_file,output_slice

  !> does nothing much now; could be used to switch processes performing i/o
  integer :: rank_write = 0

  integer, parameter :: max_luns = 64, max_filename_len=512,max_fileprop_len=32
  integer, dimension(max_luns) :: lun_list
  integer :: nluns = 0
  character (len=8) :: clun
  character, parameter :: space = ' '

  interface output_slice
    module procedure output_slice_2d
    module procedure output_slice_1d
  end interface

!  interface output_array
!    module procedure output_array_1d_complex
!  end interface

  contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> decide which processors should be responsible for writing to a file
!----------------------------------------------------------------------------

subroutine decide_who_writes(comm,lwrite,force_rank_write)

  integer, intent(in) :: comm
  logical, intent(out) :: lwrite

  integer, optional, intent(in) :: force_rank_write
  
  logical :: lwrite_all
  integer :: ierr

  ! first set lwrite to be true on a single processor
  lwrite = .false.
  if (present(force_rank_write)) then  
    if (processor_number == force_rank_write) lwrite = .true.
  else
    if (processor_number == rank_write) lwrite = .true.
  end if

  ! set lwrite on all procs with that communicator via logical .or.
#if defined(mpi)
  call MPI_ALLREDUCE(lwrite,lwrite_all,1,MPI_LOGICAL,MPI_LOR,comm,ierr)
  lwrite = lwrite_all
#endif

end subroutine decide_who_writes

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Open file wrapper; could be used for MPI I/O and HDF5 opens.
!> We (to be on the safe side) need to deal with all possible cases; write
!> them as necessary.
!----------------------------------------------------------------------------

subroutine open_file(lun,FILE,FORM,POSITION,RECL)

  integer, intent(out) :: lun
  character (len=*), intent(in) :: FILE,FORM,POSITION
  integer, intent(in) :: RECL

  optional :: FORM,POSITION,RECL
  
  character (len=max_filename_len) :: file_name
  character (len=max_fileprop_len) :: file_form,file_position

  integer :: lenf
 
  logical :: op
  
  ! unit
  lun = -i_huge
  call get_free_lun(lun)
  write(clun,'(I8.8)') lun
  
  ! name
  lenf = len(FILE)
  file_name (1:max_filename_len) = space
  file_name(1:lenf) = FILE
  
  ! form
  file_form (1:max_fileprop_len) = space
  if (present(FORM)) then
    file_form(1:len(FORM))=FORM
  else
    file_form(1:9)='formatted'
  end if
  
  ! position
  file_position(1:max_fileprop_len) = space
  if (present(POSITION)) then
    file_position(1:len(POSITION))=POSITION
  else
    file_position(1:4)='asis'
  end if
  
  if (lverbose) then
    write (*,*) '* opening file '//file_name(1:lenf)//', UNIT='//trim(clun)//','
    write (*,*) '  FORM='//trim(file_form)//', POSITION='//trim(file_position)
  end if

  ! open the file
  if (file_form(1:9) == 'formatted') then
    if (file_position(1:4) == 'asis') then
      open(lun,FILE=file_name(1:lenf),POSITION='asis')
    else if (file_position(1:6) == 'append') then
      open(lun,FILE=file_name(1:lenf),POSITION='append')
    else
      if (lverbose) then
        write(*,*) '* error opening file: '//file_name(1:lenf)//'!'
        write(*,*) '* unknown keyword option POSITION='//file_position
      end if
    end if
  else if (file_form(1:11) == 'unformatted') then
    if (present(RECL)) then
      open(lun,FILE=file_name(1:lenf),FORM='unformatted',ACCESS='direct',RECL=RECL)
    else
      open(lun,FILE=file_name(1:lenf),FORM='unformatted')      
    end if
  else
    if (lverbose) then
      write(*,*) '* error opening file: '//file_name(1:lenf)//'!'
      write(*,*) '* unknown keyword option FORM='//file_form
    end if
  end if    
  
  ! add to the list of luns which may be open
  nluns = nluns + 1
  if (nluns > max_luns) then
    if (lverbose) then
      write (*,*) '* warning: open_file: can not keep track of any more luns'
    end if
    return
  end if
  lun_list(nluns) = lun
  
end subroutine open_file

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> close file
!----------------------------------------------------------------------------

subroutine close_file(lun)

  integer, intent(in) :: lun
  integer :: i,j
  
  logical :: op

  op = .false.
  i=1
  find_lun : do
    if (lun_list(i) == lun) then
      inquire(unit=lun,opened=op)
      if (op) close(lun)
      do j=i, nluns - 1
        lun_list(j)=lun_list(j+1)
      end do
      nluns = nluns - 1
      write(clun,'(I8.8)') lun
      if (lverbose) write(*,*) '* closed file on unit '//clun
      exit find_lun
    end if
    i=i+1
    if (i > max_luns) exit find_lun
  end do find_lun

  if (.not. op) then
    inquire(unit=lun,opened=op)
    if (op) close(lun)
  end if
  
end subroutine close_file

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> close output files
!----------------------------------------------------------------------------

subroutine close_output_files
  
  integer :: i
  logical :: op
  
  do i=1,nluns
    inquire(unit=lun_list(i),opened=op)
    if (op) then
      close(lun_list(i))
      write(clun,'(I8.8)') lun_list(i)
      if (lverbose) write(*,*) '* closed file on unit '//clun
    end if
  end do

end subroutine close_output_files

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> obtain a free Logical Unit Number for input/output
!----------------------------------------------------------------------------

subroutine get_free_lun(lun)

  integer, intent(out) :: lun
  integer, parameter :: nluns = 20
  logical :: op

  lun = 110 + nluns*processor_number
  get_lun :do
    inquire(unit=lun,opened=op)
    if (.not. op) exit get_lun
    lun = lun +1
    if (lun .ge. 110+nluns*(processor_number+1)) then
       write(*,*) '*** gkw abort should be called here in get_free_lun'
      !call gkw_abort('get_free_lun: can not safely find a free lun')
       stop
    end if
  end do get_lun

end subroutine get_free_lun

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine flush_file(lun)

  integer, intent(in) :: lun
  integer :: i
  character (len=max_filename_len) :: fname
 
  if (file_is_open(lun)) then
#ifdef STD2003_FLUSH
    flush(lun)
#else
    inquire(UNIT=lun,NAME=fname)
    close(lun)
    open(lun,FILE=fname,POSITION='append')
#endif
  end if

end subroutine flush_file

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function file_is_open(lun)

  integer, intent(in) :: lun
  logical :: file_is_open
  
  if (lun <= 0) then
    file_is_open=.false.
    return
  end if
  inquire(unit=lun,opened=file_is_open)

end function file_is_open

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> output a real 2-D slice ( a(:,:) ) to file (FILE), either formatted
!> (via FMT), or as RAW binary.
!----------------------------------------------------------------------------

subroutine output_slice_2d(a,FILE,FMT)

  real, dimension(:,:), intent(in) :: a
  character (len=*), intent(in) :: FILE
  character (len=*), optional, intent(in) :: FMT
  integer :: i,j,iend,jend,lun,record_length

  ! array sizes
  iend=size(a,1) ; jend=size(a,2)

  if (present(FMT)) then
    call open_file(lun,FILE=FILE)
    do j=1,jend
      write(lun,FMT=FMT) (a(i,j), i=1,iend)
    end do
    call close_file(lun)
  else
    inquire(iolength=record_length) a
    call open_file(lun,FILE,FORM='unformatted',RECL=record_length)
    write(lun) a
    call close_file(lun)
  end if
    
end subroutine output_slice_2d

!****************************************************************************

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> output a real 1-D slice ( a(:) ) to file (FILE), either formatted
!> (via FMT), or as RAW binary.
!----------------------------------------------------------------------------

subroutine output_slice_1d(a,FILE,FMT)

  real, dimension(:), intent(in) :: a
  character (len=*), intent(in) :: FILE
  character (len=*), optional, intent(in) :: FMT
  integer :: i,j,iend,jend,lun,record_length

  ! array sizes
  iend=size(a,1)

  if (present(FMT)) then
    call open_file(lun,FILE=FILE)
      write(lun,FMT=FMT) (a(i), i=1,iend)
    call close_file(lun)
  else
    inquire(iolength=record_length) a
    call open_file(lun,FILE,FORM='unformatted',RECL=record_length)
      write(lun,REC=1) a
    call close_file(lun)
  end if
    
end subroutine output_slice_1d

!****************************************************************************
!****************************************************************************

end module io
