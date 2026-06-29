module matdat
! SVN:$Id: matdat.F90 1020 2009-07-02 19:56:45Z  $
! The module mat contains all the quantities associated
! with the matrix that contains the linear terms

implicit none

private

public :: compress_matrix, finish_matrix_section,print_matrix, iac, ii, iir
public :: iiy, iiz, jj, jjr, jjy, jjz, mat, matdat_allocate, matr, maty
public :: matz, n1, n2, n2r, n3, n3r, n4, n4r, nmat, nmaty, nmatz
public :: nmata, mata, jja 
public :: put_element, put_elem_zonal, put_source, source
public :: put_element_correct_apar

integer :: n1    = 0
integer :: n2    = 0 
integer :: n3    = 0
integer :: n4    = 0
integer :: nmat  = 0
integer :: n1r   = 0 
integer :: n2r   = 0
integer :: n3r   = 0
integer :: n4r   = 0
integer :: nmatr = 0
integer :: nmaty = 0 
integer :: nmatz = 0 
integer :: nmata = 0 

! (ntot = 25*nsp*nx*ns*nv*(nt+1))
! nsolc=nsp*nx*ns*nv*(nt+1)
! ii contains the index <-
! jj contains the index ->
! it contains the index if the term has a dt
 
integer,    allocatable :: ii(:)
integer,    allocatable :: jj(:)
complex, allocatable :: mat(:)
integer,    allocatable :: iir(:)
integer,    allocatable :: jjr(:)
real,     allocatable :: matr(:)
integer,    allocatable :: iiy(:)
integer,    allocatable :: jjy(:)
complex, allocatable :: maty(:)
integer,    allocatable :: iiz(:)
integer,    allocatable :: jjz(:)
real,     allocatable :: matz(:)
complex, allocatable :: source(:)
integer, allocatable :: jja(:)
complex, allocatable :: mata(:) 

! integeter array for the implicit scheme 
integer,    allocatable :: iac(:)

save :: n1, n2, n3, n4, nmat, n1r, n2r, n3r, n4r, nmatr, ii, jj, mat

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine matdat_allocate
!--------------------------------------------------------------------
! This routine allocates the arrays of the module matdat
!--------------------------------------------------------------------
use dist,    only : nsolc, ntot, nphi
use control, only : matrix_format, method, zonal_adiabatic, nlapar
use grid,    only : nx, ns, nsp
! local variables 
integer ierr
integer i 

! initialize the error parameter 
ierr = 0

! allocate the source array 
allocate(source(nsolc),stat=ierr)
if (ierr.ne.0) then 
  stop 'Could not allocate source in matdat'
endif
! intialize the source to zero 
do i = 1, nsolc 
  source(i) = (0.,0.)
end do 
if (zonal_adiabatic) then 
  allocate(iiy(nx),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate iiz in matdat'
  endif
  allocate(jjy(nx),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate jjz in matdat'
  endif
  allocate(maty(nx),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate matz in matdat'
  endif 
  allocate(iiz(nx*ns),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate iiz in matdat'
  endif
  allocate(jjz(nx*ns),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate jjz in matdat'
  endif
  allocate(matz(nx*ns),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate matz in matdat'
  endif
endif 


select case(matrix_format)

case('complex')

  ! allocate the index array
  allocate(ii(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate ii in matdat'
  endif

  ! allocate the index array
  allocate(jj(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate jj in matdat' 
  endif

  ! allocate the matrix
  allocate(mat(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate mat in matdat'
  endif

  ! the arrays for the correction due to A||
  if (nlapar) then 

    allocate(jja(nsolc), stat = ierr) 
    if (ierr.ne.0) then 
      stop 'Could not allocate jja in matdat'
    endif 

    allocate(mata(nsolc), stat = ierr) 
    if (ierr.ne.0) then 
      stop 'Could not allocate mata in matdat'
    endif 

  endif 
 
case('complex-real')

  ! allocate the index array
  allocate(ii(4*nsolc),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate ii in matdat'
  endif

  ! allocate the index array
  allocate(jj(4*nsolc),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate jj in matdat' 
  endif

  ! allocate the matrix
  allocate(mat(4*nsolc),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate mat in matdat'
  endif

  ! allocate the index array
  allocate(iir(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate ii in matdat'
  endif

  ! allocate the index array
  allocate(jjr(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate jj in matdat' 
  endif

  ! allocate the matrix
  allocate(matr(ntot),stat=ierr)
  if (ierr.ne.0) then 
    stop 'Could not allocate matr in matdat'
  endif

case default 

  stop 'Severe internal error unkown matrix format'

end select 

allocate(iac(nsolc+1),stat=ierr)
if (ierr.ne.0) then 
  stop 'Cannot allocate iac in compress_matrix'
endif


return
end subroutine matdat_allocate

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine put_element(iih,jjh,mat_elem,itime_est)
!--------------------------------------------------------------------
! This routine put an element in the matrix 
! iih      -> i index 
! jjh      -> j index 
! mat_elem -> the matrix element
! The routine checks if ntot is not exceeded 
!-------------------------------------------------------------------- 
use dist,    only : ntot, nsolc
use control, only : matrix_format, collisions
use general, only : gkw_warn, time_est

integer,    intent(in) :: iih, jjh
integer, optional, intent(in) :: itime_est
complex, intent(in) :: mat_elem

logical, save :: initialised = .FALSE.

if (.not. initialised) then 
  initialised = .TRUE. 
  nmat  = 0
  nmatr = 0
endif

! different formats of the matrix 
select case(matrix_format)

case('complex')

  ! check for the size of the matrix 
  if (nmat.ge.ntot) stop 'Not enough elements to store the matrix'

  ! put the element
  if (abs(mat_elem) .gt. 1.E10 .and. collisions) then
    call gkw_warn('put_element: rejecting large element for collisions')
    return
  endif 
  nmat = nmat + 1
  ii(nmat) = iih
  jj(nmat) = jjh
  mat(nmat) = mat_elem  

case('complex-real')

  if (aimag(mat_elem).eq.0) then 
   
    ! check for the size of the matrix 
    if (nmatr.ge.ntot) stop 'Not enough elements to store the matrix'
     
    ! put the element in the real matrix 
    nmatr = nmatr + 1 
    iir(nmatr) = iih
    jjr(nmatr) = jjh 
    matr(nmatr) = real(mat_elem) 
  
  else 
 
    ! check for the size of the matrix 
    if (nmat.ge.4*nsolc) stop 'Not enough elements to store the matrix' 
 
    ! put the element in the complex matrix
    nmat = nmat + 1 
    ii(nmat) = iih
    jj(nmat) = jjh 
    mat(nmat) = mat_elem
        
  endif 
    
case default 

  stop 'Severe internal error, unkown matrix format'

end select
 
!if (present(itime_est)) call time_est(mat_elem,itime_est)

! always call the time step estimator
call time_est(mat_elem,88)

end subroutine put_element 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine put_elem_zonal(iih,jjh,mat_elem,isw) 
!--------------------------------------------------------------------
! put the matrix element for the zonal flow computation 
!--------------------------------------------------------------------
use grid, only : nx, ns, nsp 

implicit none 

integer,    intent(in) :: iih, jjh, isw 
complex, intent(in) :: mat_elem

select case(isw)

case(1)

  ! put the element in the matrix 
  nmatz= nmatz + 1 

  if (nmatz.gt.nx*ns) then 
    write(*,*)'Not enough elements for the zonal matrix' 
    stop 
  endif 

  iiz(nmatz) = iih
  jjz(nmatz) = jjh 
  matz(nmatz) = real(mat_elem) 

  return 

case(2) 

  ! put the average element 
  nmaty = nmaty + 1 

  if (nmaty.gt.nx) then 
    write(*,*)'Not enough elements for the zonal matrix'
    stop
  endif 

  iiy(nmaty) = iih 
  jjy(nmaty) = jjh
  maty(nmaty) = real(mat_elem) 

  return 

case default 

  write(*,*)'Routine put_elem_zonal called with wrong isw'
  stop 

end select 

return 
end subroutine put_elem_zonal 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine put_element_correct_apar(jjh,mat_elem) 
!--------------------------------------------------------------------
! This subroutine puts the elements of the matrix that are used for 
! the A|| correction in the distribution function 
!--------------------------------------------------------------------
use dist, only : nsolc
use general, only : gkw_abort

implicit none 

integer, intent(in) :: jjh 
complex, intent(in) :: mat_elem 

nmata = nmata + 1 
if (nmata.gt.nsolc) call gkw_abort('MATDAT put_element_correct_apar: &
                                   &Too many elements in mata') 

jja(nmata) = jjh 
mata(nmata) = mat_elem 

return
end subroutine 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine put_source(iih,mat_elem) 
!------------------------------------------------------------------------------
! This routine stores the source 
!------------------------------------------------------------------------------
implicit none 

integer,    intent(in) :: iih 
complex, intent(in) :: mat_elem 

source(iih) = source(iih) + mat_elem 

return 
end subroutine put_source 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine finish_matrix_section(isel) 

implicit none 

integer, intent(in) :: isel 

select case(isel)

case(1) 
  n1  = nmat
  n1r = nmatr
case(2)
  n2  = nmat
  n2r = nmatr
case(3)
  n3  = nmat
  n3r = nmatr 
case(4) 
  n4  = nmat 
  n4r = nmatr 
case default 
  stop 'isel out of range in finish_matrix_selection. Panick!'
end select

end subroutine finish_matrix_section

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine compress_matrix
!--------------------------------------------------------------------
! This routine call the compression routine which orders the 
! elements and adds double elements in the matrix 
!--------------------------------------------------------------------
use dist,            only : fdisi, nsolc, msolc, ntot, indx, nphi, iphi
use control,         only : method, dtim, silent, matrix_format
use grid,            only : nx, ns, nsp
use general,         only : gkw_abort  

#if defined(mpi)
use mpiinterface
#endif 

implicit none 

integer j, nt1, nt2, itel, i, np, ierr, k, ix, is, &
      & ireduced, nelem
complex w
   
integer, allocatable :: iwksp(:)

! Some simple tests on the matrix 
if (n4.gt.ntot) stop 'n4 too large'
do i = 1, n4
  if ((ii(i).gt.nsolc).or.(ii(i).lt.1)) then 
    call gkw_abort('ii - Outside of range in compress_matrix')
  endif 
  if ((jj(i).gt.msolc).or.(jj(i).lt.1)) then 
    call gkw_abort('jj - Outside of range in compress_matrix')
  endif 
end do 


! The ordering of the matrix depends a little on the method. 
select case(method)

  case('EXP')

    select case(matrix_format) 

      case('complex')

        ! Three parts are separately surpressed. Note there is 
        ! anyway no overlap
        ireduced = 1
        call compress_section(1)

      case('complex-real')

        ! Three parts are separately surpressed. Note there is 
        ! anyway no overlap
        ireduced = 1
        call compress_section(1)
 
      case default 

        call gkw_abort('Severe internal error, unkown matrix format')

    end select 

    ! set up an ia array
    iac(1) = 1
    j = 1
    do i = 1, nsolc-1
      659 continue
      if (ii(j).eq.i) then 
        j = j+1
        if (j.le.n4) goto 659
      endif
      iac(i+1) = j
    end do 
    iac(nsolc+1) = n3+1

    return

  case('IMP')

#if defined(umfpack) 

    ! umfpack has the opposite definition for the 
    ! matrix format. This is solved her through 
    ! the exchange of ii and jj 
    do i = 1, n4 
      ireduced = ii(i) 
      ii(i) = jj(i) 
      jj(i) = ireduced 
    end do 

#endif 

    ! compress all the matrix 
    ireduced = 1
    call compress_section(0)


#if defined(umfpack) 

    ! umfpack has the opposite definition for the 
    ! matrix format. This is solved her through 
    ! the exchange of ii and jj 
    do i = 1, n4 
      ireduced = ii(i) 
      ii(i) = jj(i) 
      jj(i) = ireduced 
    end do 

#endif 

    ! do a test on the array
    allocate(iwksp(nsolc),stat=ierr)
    if (ierr /= 0) call gkw_abort('Cannot allocate iwksp in compress_matrix')
    do i = 1, nsolc
      iwksp(i) = 0
    end do 
    do i = 1, n4
      iwksp(ii(i)) = 1
    end do 
    do i = 1, nsolc
      if (iwksp(i).eq.0) then
        write(*,*)'Element ',i,' is not referenced' 
        stop
      endif
    end do 
    deallocate(iwksp,stat=ierr)
    if (ierr /= 0) call gkw_abort('Cannot deallocate iwksp in compress_matrix')


#if defined(umfpack)

    ! set up an ia array 
    iac(1) = 1 
    i = 1 
    do j = 1, nsolc-1 
      649 continue 
      if (jj(i).eq.j) then 
        i = i + 1 
        if (i.le.n4) goto 649 
      endif 
      iac(j+1) = i 
    end do 
    iac(nsolc+1) = n4+1 

#else 

    ! set up an ia array
    iac(1) = 1
    j = 1
    do i = 1, nsolc-1
      658 continue
      if (ii(j).eq.i) then 
        j = j+1
        if (j.le.n4) goto 658
      endif
      iac(i+1) = j
    end do 
    iac(nsolc+1) = n4+1

#endif 

    return

  case('EGV')

    ! look in the old routine to merge this again into the 
    ! solution 
    call gkw_abort('Not implemented in this version') 
 
  case default 

    !Error 
    call gkw_abort('Unknown numerical scheme')

end select 

end subroutine compress_matrix 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine compress_section(isel)
! routine cleans up (compresses part of the matrix). 
! nelem : number of elements 
! istart : start compression from 
! indd, iwksp wskp  workspace arrays 
! ireduced number of elements after compression, Note. The matrix
! is more or less destroyed if one does a small block compression
! several sequential calls are required. that complete the whole 
! matrix for the routine to work properly. Note ireduced is input 
! and output. Set it to 1 at first call. Do not change it in 
! beween 

use control, only : matrix_format, silent 
use dist,    only : ntot 

implicit none 

integer isel ! switch 0 : whole matrix is compressed 
             !        1 : the three different sections are 
             !            compressed seprately 

!integer indd(ntot), iwksp(ntot)
!complex wksp(ntot) 

integer i, j, ibegin, istart, nelem, ireduced
integer nt1, nt2, nt3, nt4, iloop, ilpm, itmp

ireduced = 1 

ilpm = 1 
if (isel.eq.1) ilpm = 4 
if (isel.eq.2) ilpm = 3 

do iloop = 1, ilpm   

  select case(isel) 
  case(0) 
    istart = 1 
    nelem  = n4
  case(1)
    select case(iloop) 
    case(1) 
      istart = 1 
      nelem  = n1 
    case(2)
      istart = n1+1
      nelem = n2 -n1 
    case(3) 
      istart = n2 + 1 
      nelem  = n3 - n2 
    case(4)
      istart = n3 + 1 
      nelem  = n4 - n3
    case default 
      stop 'Error iloop out or range' 
    end select 
  case(2) 
    select case(iloop) 
    case(1)
      istart = 1 
      nelem = n2 
    case(2) 
      istart = n2+1 
      nelem  = n3 - n2 
    case(3) 
      istart = n3+1 
      nelem = n4 - n3 
    end select 
  case default 
    stop 'Severe internal error, isel out of range'
  end select
     
  ! sort the matrix  
   call slow_heap_sort(nelem,istart)

  ! select the proper start of the compression loop 
  if (ireduced.gt.istart) stop 'ireduced larger istart'
  if (ireduced.eq.istart) then 
    ibegin = istart + 1
  else 
    ibegin = istart 
  endif 

  ! compress the matrix 
  do i = ibegin, istart+nelem-1
    if ((ii(i).eq.ii(ireduced)).and.(jj(i).eq.jj(ireduced))) then
      mat(ireduced) = mat(ireduced) + mat(i)
    else
      ireduced = ireduced + 1
      ii(ireduced) = ii(i)
      jj(ireduced) = jj(i)
      mat(ireduced) = mat(i)
    endif  
  end do 

  select case(iloop) 
    case(1) 
      nt1 = ireduced 
    case(2) 
      nt2 = ireduced 
    case(3)
      nt3 = ireduced 
    case(4) 
      nt4 = ireduced
    case default 
      stop 'iloop out of range'
  end select 
end do 

select case(isel) 

case(0)   
 
  if (.not.silent) then 
    write(*,*)'******************************************************'
    write(*,*)'Matrix compression successfully completed'
    write(*,223)n4,nt1
    223  format(' Original ',I8,' elements. New ',I8,' elements') 
    write(*,*)'******************************************************'
    write(*,*)
  endif 
  n1 = nt1
  n2 = nt1
  n3 = nt1 
  n4 = nt1
 
case(1) 
 
  if (.not.silent) then 
    write(*,*)'******************************************************'
    write(*,*)'Matrix compression successfully completed'
    write(*,223)n4,nt4
    write(*,*)'******************************************************'
    write(*,*)
  endif 
  n1 = nt1
  n2 = nt2 
  n3 = nt3
  n4 = nt4 

case(2) 

  if (.not.silent) then 
    write(*,*)'******************************************************'
    write(*,*)'Matrix compression successfully completed'
    write(*,223)n3,nt3
    write(*,*)'******************************************************'
    write(*,*)
  endif 
  n1 = nt1 
  n2 = nt1 
  n3 = nt2 
  n4 = nt3  

case default 
 
  stop 'isel out of range'
 
end select 

if (matrix_format.eq.'complex-real') then 
! in this case also the real matrix must be compressed.  

ireduced = 1 

do iloop = 1, ilpm  

  select case(isel) 
  case(0) 
    istart = 1 
    nelem  = n4r
  case(1)
    select case(iloop) 
    case(1) 
      istart = 1 
      nelem  = n1r 
    case(2)
      istart = n1r+1
      nelem = n2r -n1r 
    case(3) 
      istart = n2r + 1 
      nelem  = n3r - n2r 
    case(4) 
      istart = n3r + 1 
      nelem  = n4r - n3r 
    case default 
      stop 'Error iloop out or range' 
    end select 
  case default 
    stop 'Severe internal error, isel out of range'
  end select
     
  ! sort the matrix  
  call slow_heap_sort(nelem,istart)

  ! select the proper start of the compression loop 
  if (ireduced.gt.istart) stop 'ireduced larger istart'
  if (ireduced.eq.istart) then 
    ibegin = istart + 1
  else 
    ibegin = istart 
  endif 

  ! compress the matrix 
  do i = ibegin, istart+nelem-1
    if ((iir(i).eq.iir(ireduced)).and.(jjr(i).eq.jjr(ireduced))) then
      matr(ireduced) = matr(ireduced) + matr(i)
    else
      ireduced = ireduced + 1
      iir(ireduced) = iir(i)
      jjr(ireduced) = jjr(i)
      matr(ireduced) = matr(i)
    endif  
  end do 

  select case(iloop) 
    case(1) 
      nt1 = ireduced 
    case(2) 
      nt2 = ireduced 
    case(3)
      nt3 = ireduced 
    case(4) 
      nt4 = ireduced 
    case default 
      stop 'iloop out of range'
  end select 
end do 

select case(isel) 

case(0)   
 
  if (.not.silent) then 
    write(*,*)'******************************************************'
    write(*,*)'Matrix compression successfully completed'
    write(*,223)n4r,nt1 
    write(*,*)'******************************************************'
    write(*,*)
  endif 
  n1r = nt1
  n2r = nt1
  n3r = nt1 
  n4r = nt1 
 
case(1) 
 
  if (.not.silent) then 
    write(*,*)'******************************************************'
    write(*,*)'Matrix compression successfully completed'
    write(*,223)n4r,nt4
    write(*,*)'******************************************************'
    write(*,*)
  endif 
  n1r = nt1
  n2r = nt2 
  n3r = nt3
  n4r = nt4 
 
case default 
 
  stop 'isel out of range'
 
end select 

endif 

return 
end subroutine compress_section 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine slow_heap_sort(n_elem,i_start)

  integer, intent(in) :: n_elem,i_start
  integer :: itmp,i,ind
  complex :: ctmp

  do i=(n_elem/2) - 1, 0, -1
    call siftd(i,n_elem,i_start)
  end do

  do i=n_elem - 1, 1, -1
    ind  = i + i_start
    itmp = ii(ind) ; ii(ind) = ii(i_start) ; ii(i_start) = itmp
    itmp = jj(ind) ; jj(ind) = jj(i_start) ; jj(i_start) = itmp
    ctmp = mat(ind) ; mat(ind) = mat(i_start) ; mat(i_start) = ctmp
    call siftd(0,i-1,i_start)
  end do
 
end subroutine slow_heap_sort

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine siftd(b,e,iii)

  integer, intent(in) :: b,e,iii
  integer :: c, root, itmp,ind,ind2
  complex :: ctmp

  root = b
  rootsub : do

    c = 2*root
    if (c > e) exit rootsub
    
    ind = c + iii
    if (2*root == e .or. larger(ii(ind),jj(ind),ii(ind+1),jj(ind+1))) then
      ! do nothing
    else
      c    = c + 1
      ind  = ind + 1
    end if

    ind2 = root + iii
    if (less(ii(ind2),jj(ind2),ii(ind),jj(ind))) then
      itmp = ii(ind2) ; ii(ind2) = ii(ind) ; ii(ind) = itmp
      itmp = jj(ind2) ; jj(ind2) = jj(ind) ; jj(ind) = itmp
      ctmp = mat(ind2) ; mat(ind2) = mat(ind) ; mat(ind) = ctmp
      root = c
    else
      exit rootsub
    end if

  end do rootsub

end subroutine siftd

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function larger(ii1,jj1,ii2,jj2)

  integer, intent(in) :: ii1, ii2, jj1, jj2
  logical :: larger

  larger = (ii1 > ii2) .or. (ii1 == ii2 .and. jj1 > jj2)

end function larger 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function lesseq(ii1,jj1,ii2,jj2)

  integer, intent(in) :: ii1, ii2, jj1, jj2
  logical :: lesseq

  lesseq =  (ii1 < ii2) .or. (ii1 == ii2 .and. jj1 <= jj2)

end function lesseq

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function less(ii1,jj1,ii2,jj2)

  integer, intent(in) :: ii1, ii2, jj1, jj2
  logical :: less

  less = (ii1 < ii2) .or. (ii1 == ii2 .and. jj1 < jj2)

end function less

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine print_matrix
use dist,            only : fdisi, nsolc, ntot
integer :: i
real :: re, im

open(18,file = 'full_matrix.dat') !The grid at the high field point
do i=1,ntot
   re = real(mat(i))
   im = aimag(mat(i))
   write(18,*)ii(i),jj(i),re,im
end do
close(18)
!stop 'Thats all i want'
return 

end subroutine print_matrix

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module matdat

