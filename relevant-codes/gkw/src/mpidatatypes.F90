!----------------------------------------------------------------------------
!> $Id: mpidatatypes.F90 1005 2009-07-02 16:12:03Z  $
!! provide derived datatypes for communication of processor boundaries and
!< parallel io etc.
!----------------------------------------------------------------------------

module mpidatatypes
  
  implicit none

  private

  ! --- Public Routines ---
  public :: create_subarray_datatype,create_fdisi_io_type
  
  ! --- Public --- !
  integer, public :: TYPE_NEXT_MU
  integer, public :: TYPE_PREV_MU
  integer, public :: TYPE_NEXT_VPAR
  integer, public :: TYPE_PREV_VPAR
  integer, public :: TYPE_NEXT_S
  integer, public :: TYPE_PREV_S
  integer, public :: TYPE_NEXT_S_NEXT_VPAR
  integer, public :: TYPE_NEXT_S_PREV_VPAR
  integer, public :: TYPE_PREV_S_NEXT_VPAR
  integer, public :: TYPE_PREV_S_PREV_VPAR
  integer, public :: TYPE_NEXT_VPAR_NEXT_MU
  integer, public :: TYPE_PREV_VPAR_NEXT_MU
  integer, public :: TYPE_NEXT_VPAR_PREV_MU
  integer, public :: TYPE_PREV_VPAR_PREV_MU 
  integer, public :: TYPE_RW_FDISI          !< fdisi to/from file

contains

!****************************************************************************
!****************************************************************************
!> Create a new sub-array datatype given an existing datatype and up to 6
!> identifiers for the various dimensions (found in the global module, all
!> prefixed `id_'). Typically, the new datatype is used to write a local array
!> through a `slot' into a global array on file. This is done for the restart
!> file where the local processor has a sub-array of the global solution.
!----------------------------------------------------------------------------
  
subroutine create_subarray_datatype(old_type,new_type,i1,i2,i3,i4,i5,i6)

  use mpiinterface
  use general, only : gkw_abort
  use global,  only : id_s,id_sp,id_vpar,id_mu,id_x,id_mod,id_dummy
  use grid,    only : n_x_grid,nmod,n_s_grid,ns,n_mu_grid,nmu,n_vpar_grid,nx,&
                    & nvpar,number_of_species,nsp,ivparpb,isppb,imupb,ispb,ixpb

  integer, intent(in) :: old_type
  integer, intent(out) :: new_type
  integer, intent(in) :: i1
  integer, optional, intent(in) :: i2,i3,i4,i5,i6
 
  integer :: ndims,i,ierr
  integer, parameter :: maxdims=6
  integer, dimension(maxdims) :: gs,ls,st,id_in
  logical :: ldum
  
  id_in(:) = id_dummy
  id_in(1) = i1
  if (present(i2)) id_in(2) = i2
  if (present(i3)) id_in(3) = i3
  if (present(i4)) id_in(4) = i4
  if (present(i5)) id_in(5) = i5
  if (present(i6)) id_in(6) = i6
  
  ndims = 0
  get_dims : do i=1, maxdims
  
    ldum = .false.
    select case (id_in(i))
      case(id_s)    ; gs(i)=n_s_grid          ; ls(i)=ns    ; st(i)=ispb-1
      case(id_vpar) ; gs(i)=n_vpar_grid       ; ls(i)=nvpar ; st(i)=ivparpb-1
      case(id_mu)   ; gs(i)=n_mu_grid         ; ls(i)=nmu   ; st(i)=imupb-1
      case(id_x)    ; gs(i)=n_x_grid          ; ls(i)=nx    ; st(i)=ixpb-1
      case(id_mod)  ; gs(i)=nmod              ; ls(i)=nmod  ; st(i)=0
      case(id_sp)   ; gs(i)=number_of_species ; ls(i)=nsp   ; st(i)=isppb-1
      case(id_dummy)
        ldum = .true.
      case default
        call gkw_abort('bad id in mpidatatypes')
    end select
    
    if (ldum) exit get_dims
    ndims=ndims+1

  end do get_dims

  ! create the subarray datatype
#ifdef mpi
  ierr=0
  call MPI_TYPE_CREATE_SUBARRAY(ndims,gs,ls,st,MPI_ORDER_FORTRAN,old_type,   &
      &  new_type,ierr)
  call MPI_TYPE_COMMIT(new_type,ierr)
#else
  new_type = 0
#endif

end subroutine create_subarray_datatype

!****************************************************************************
!> Create a datatype for reading/writing the distribution function from/to
!! file in a particular order. The distribution function can be stored in
!! any order in memory, but for the purpose of restarting a run with an
!! alternative memory layout it is useful to always have the file data in the
!! same order. The order of the data in the file could also be recorded in the
!! file so that it can always be correctly read, if for some reason is useful
!! to have differently ordered outputs.
!!
!! Presently the input `type_old' for this routine is MPICOMPLEX_X and the
!! returned type `type_new' is the then the type to write a the full local
!! This is particularly useful for writing the distribution function
!! APS: subroutine get_reordered_type(ndims,dims,type_in,type_out)
!< returns old_type for non-MPI run
!----------------------------------------------------------------------------

subroutine create_fdisi_io_type

  use global
  use general, only : gkw_abort
  use index_function
  use mpiinterface
  use grid, only : nsp,nvpar,nmu,ns,nx,nmod

  integer :: nf
  
  integer, allocatable, dimension(:) :: offset
  integer :: blocklen
  integer, dimension(6) :: iend
  integer :: ierr, new_index
  integer :: ispecies, ivpar,imu,is,ix,imod
  integer :: i,j,k,l,m,n
  integer :: u_count,TYPE_OLD

  u_count = 0
  blocklen = 1
  TYPE_OLD=MPICOMPLEX_X
  nf=nsp*nvpar*nmu*ns*nx*nmod
  allocate(offset(nf),stat=ierr) 
  if (ierr /= 0) call gkw_abort('get_reordered_type: cannot allocate offset')

!  allocate(blocklen(nf),stat=ierr)
!  if (ierr /= 0) call gkw_abort('get_reordered_type: cannot allocate blocklen')

  iend(:) = 1
!APS  iend(1:n_dims) = dims(1:n_dims)
  iend(1) = nsp
  iend(2) = nmod
  iend(3) = nx
  iend(4) = ns
  iend(5) = nmu
  iend(6) = nvpar
  
  new_index = 0
  do n= 1, iend(1)
    do m= 1, iend(2)
      do l= 1, iend(3)
        do k= 1, iend(4)
          do j = 1, iend(5)
            do i = 1, iend(6)
              !
              ispecies = n
              imod     = m
              ix       = l
              is       = k
              imu      = j
              ivpar    = i
              !ivpar,imu,is,ix,imod,isp
              new_index = new_index + 1
              ! starts from zero in mpi call

              offset(new_index) = indx_(imod, ix, is, imu, ivpar, ispecies) - 1
              if (offset(new_index) + 1 == new_index) u_count = u_count + 1
              !blocklen(new_index) = 1
            end do
          end do
        end do
      end do
    end do
  end do

  if (.not. u_count == nf) then
    if (lverbose) write (*,*) '* re-ordered memory layout'
#if defined(mpi)
    call MPI_TYPE_CREATE_INDEXED_BLOCK(nf,blocklen,offset,TYPE_OLD,          &
        &    TYPE_RW_FDISI,ierr)
#endif
  else
    if (lverbose) write (*,*) '* default memory layout'
#if defined(mpi)
    call MPI_TYPE_CONTIGUOUS(nf,TYPE_OLD,TYPE_RW_FDISI,ierr)
#endif
  end if
  
#if defined(mpi)
  call MPI_TYPE_COMMIT(TYPE_RW_FDISI,ierr)
#else
  ! not very good, but there are no obvious better choices
  TYPE_RW_FDISI = TYPE_OLD
#endif

  deallocate(offset)
 ! deallocate(blocklen)
  
end subroutine create_fdisi_io_type

!**************************************************************************** 
!****************************************************************************

end module mpidatatypes
