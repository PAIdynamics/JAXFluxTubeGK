!****************************************************************************
! $Id: dist.F90 1020 2009-07-02 19:56:45Z  $
!!>
!! This module contains the code solution and sets up how the different parts
!! are arranged in the main array. The index function module is responsible
!! for the mapping between points in the comutational domain and the array
!! index and so needs to be initialised accordingly. Communicators used for
!! sending parts of the local array to other parts of the same array on other
!! processors are also initialised from here.
!<
!****************************************************************************

module dist

  use mpiinterface
  use mpidatatypes

  implicit none

  private

  !
  ! public subroutines and functions
  !

  public :: dist_init
  public :: get_apar, get_phi, get_bpar, indx

  !
  ! public parameters
  !
  
  ! parameters to reference part of the index function
  integer, public, parameter :: iphi   = -45236   !< potential index
  integer, public, parameter :: iapar  = -95321   !< apar index
  integer, public, parameter :: ibpar  = -392855  !< bpar index
  integer, public, parameter :: i_mom  = -53456   !< momentum change index
  !> For adiabatic species - this is permantly set to 1
  integer, public, parameter :: iadia = 1

  !
  ! public variables
  !

  !> The arrays to keep track of the phase for the mode
  !> frequency calculation (in normalize.f90) 
  real, public, allocatable :: phase(:,:)
  real, public, allocatable :: last_phase(:,:)

  !> The matrix has a maximum of ntot elements  
  integer, public :: ntot
  !> The distribution function (NOT including the potential) has a
  !> total of nf elements per processor.
  integer, public :: nf
  !> The distribution function (including the potential) has a total
  !> of nsolc ( = nf + nfields ) elements to solve for per processor
  integer, public :: nsolc
  !> The distribution (including the potential) has a total of 
  !> msolc ( = nsolc + 2*nghost ) elements.
  integer, public :: msolc
  !> location of the last point of `regular' fields in fdisi
  integer, public :: nregular_fields_end
  !> the number of elements in the momentum conserving quantity
  integer, public :: nelem_mom_conserve
  !> the offset for the momentum conserving quantity for collisions
  !> (from the start of the main array)
  integer, public :: n_mom_conserve
  !> the maxwell fmaxwl(ns,nmu,nvp)
  real, public, allocatable :: fmaxwl(:,:,:)
  !> the slowing down alpha particle distribution 
  real, public, allocatable :: falpha(:,:,:)
  !> the distribution fdisi(msolc)
  complex, public, allocatable :: fdisi(:)
  !> potential phi(nmod,nx,ns)
  complex, public, allocatable :: phi(:,:,:)
  !> The parallel component of the vector potential  apar(nmod,nx,ns)
  complex, public, allocatable :: apar(:,:,:)
  !> The parallel component of the magnetic field perturbation bpar(nmod,nx,ns)
  complex, public, allocatable :: bpar(:,:,:)
  !> The position where phi starts in the solution 
  integer, public :: nphi
  !> The total number of points to be communicated in some direction.
  integer, public :: ghost_size_vpar, ghost_size_s, ghost_size_mu
  integer, public :: ghost_size_vpar_mu, ghost_size_vpar_s
  integer, public :: ighost_sbp, ighost_sbn
  integer, public :: ighost_vparbp, ighost_vparbn
  integer, public :: ighost_mubp, ighost_mubn
  integer, public :: ighost_vparbp_mubp, ighost_vparbn_mubp
  integer, public :: ighost_vparbp_mubn, ighost_vparbn_mubn
  integer, public :: ighost_vparbp_sbp, ighost_vparbn_sbp
  integer, public :: ighost_vparbp_sbn, ighost_vparbn_sbn

  !needs to be public for testing
  integer, public :: n_bpar

  !
  ! private variables
  !

  !> to check in initialization routines have been called
  logical :: first_call = .true.
  !> The number of ghost points in various directions (at each end).
  integer :: ghost_points_vpar, ghost_points_s, ghost_points_mu
  integer :: ghost_points_vpar_mu, ghost_points_vpar_s
  !> flag to control how the index function works
  logical :: lindex_vpar_last
  ! various offsets
  integer :: ioffset_phi, ioffset_apar, ioffset_bpar, &
    &    n_apar, n_phi, &!n_bpar,     &
    &    nelem_apar, nelem_phi, nelem_bpar
  !> number of fields 
  integer :: nfields

  !
  ! interfaces
  !

  interface indx
    module procedure indx_main
    module procedure indx_other
  end interface

contains

!****************************************************************************
!****************************************************************************
!> subroutine that intializes everything required to use dist
!----------------------------------------------------------------------------

subroutine dist_init

  use general,        only : gkw_abort
  use index_function, only : register_offset, index_init,                    &
                           & index_set_ghostpoints, indx_
  use grid,           only : nsp, nx, ns, nmu, nvpar, nmod, parallel_vpar,   &
                           & parallel_s, lsendrecv_mu
  use control,        only : nlphi, nlapar, nlbpar, vp_trap,                 &
                           & order_of_the_scheme, collisions,                &
                           & ltrapping_arakawa

  ! we have got here into dist, so at least an attempt was made to call these
  first_call = .false.
  
  ! Set the array sizes of the matrix and index arrays 
  ntot = 64*nsp*nx*ns*nmu*(nvpar+1)*nmod

  ! The size of the solution, without any fields
  nf = nsp*nx*ns*nmu*nvpar*nmod

  ! the (initial) size of the full solution
  nsolc = nf
  nfields = 0

  !
  ! (1) Work out how many grid points to communicate between adjacent
  !     processors. This depends on the order of the scheme.
  !
  
  ghost_points_vpar = 0
  ghost_points_s = 0
  ghost_points_mu = 0
  ghost_points_vpar_mu = 0
  ghost_points_vpar_s  = 0
  ghost_points : if ((parallel_vpar .and. vp_trap == 0) .or. parallel_s .or. &
      & lsendrecv_mu) then
    select case(order_of_the_scheme)
      case('second_order')
        if (parallel_s)    ghost_points_s = 1
        if (parallel_vpar) ghost_points_vpar = 1
        if (lsendrecv_mu)  ghost_points_mu = 1
        if (parallel_vpar .and. lsendrecv_mu) ghost_points_vpar_mu = 1
        if (parallel_s    .and. parallel_vpar .and. ltrapping_arakawa) then
          ghost_points_vpar_s = 1
        end if
      case('fourth_order')
        if (parallel_s)    ghost_points_s = 2
        if (parallel_vpar) ghost_points_vpar = 2
        ! We only have second order derivatives in mu so far.
        if (lsendrecv_mu)  ghost_points_mu = 1
        ! only 1 point needed below
        if (parallel_vpar .and. lsendrecv_mu) ghost_points_vpar_mu = 1
        ! only 1 point needed below
        if (parallel_s    .and. parallel_vpar .and. ltrapping_arakawa) then
          ghost_points_vpar_s = 1
        end if
      case default
        call gkw_abort('dist_init: this should not happen')
    end select
  end if ghost_points
  
  !
  ! (2) Work out the total number of elements of complex datatype to be sent
  !     to the adjacent processors based on the required derivatives.
  !     Additional contributions will come from the fields.
  !
  
  ghost_size_vpar    = ghost_points_vpar*nsp*nx*ns*nmu*nmod
  ghost_size_mu      = ghost_points_mu*nsp*nx*ns*nvpar*nmod
  ghost_size_s       = ghost_points_s*nsp*nx*nvpar*nmu*nmod
  ghost_size_vpar_mu = ghost_points_vpar_mu*nsp*nx*ns*nmod
  ghost_size_vpar_s  = ghost_points_vpar_s*nmu*nsp*nx*nmod

  !
  ! (3) Add the fields and their contributions to the ghost elements.
  !
  
  ! if the electro-static potential is kept increase the size
  if (nlphi) then
    n_phi       = nf + nfields         ! the phi offset (=nf)
    nelem_phi   = nx*ns*nmod           ! number of elements  
    nfields     = nfields + nelem_phi  ! total size of fields
    ! the offset of phi within the ghost block
    ioffset_phi = ghost_size_s
    ! increase the ghost size; derivatives in phi are of same order as fdisi
    ghost_size_s = ghost_size_s + nx*nmod*ghost_points_s
  endif 
  
  ! if the parallel vector potential is kept increase the size
  if (nlapar) then
    n_apar      = nf + nfields
    nelem_apar  = nx*ns*nmod
    nfields     = nfields + nelem_apar
    ioffset_apar = ghost_size_s
    ghost_size_s = ghost_size_s + nx*nmod*ghost_points_s
  endif 
  
  ! if the parallel magnetic field is kept increase the size
  if (nlbpar) then
    n_bpar      = nf + nfields
    nelem_bpar  = nx*ns*nmod
    nfields     = nfields + nelem_bpar
    ioffset_bpar = ghost_size_s
    ghost_size_s = ghost_size_s + nx*nmod*ghost_points_s
  endif 

  ! N.B. no more actual fields are allowed after here.
  nregular_fields_end = nsolc + nfields
  
  ! The momentum conserving `field': no derivatives.
  if (collisions) then
    n_mom_conserve = nf + nfields
    nelem_mom_conserve = nx*nmod*ns*nsp
    nfields = nfields + nelem_mom_conserve
  end if
 
  ! increase the size of nsolc by the size of the fields
  nsolc = nsolc + nfields

  !
  ! (4) Decide on how the index function and communication will work. N.B.
  !     this is obsolete/unnecessary as the index function can work with
  !     any ordering; the most appropriate orderings should be worked out.
  !
  
  if (parallel_s) then
    if (parallel_vpar .and. (vp_trap == 0)) then
      lindex_vpar_last = .true.
    else
      lindex_vpar_last = .false.
    end if
  else
    if (parallel_vpar) then
      lindex_vpar_last = .true.
    else
      lindex_vpar_last = .false.
    end if
  end if
  
  !
  ! (5) Decide where to store the points received from adjacent processors.
  !
  
  ! Start at nsolc
  ighost_vparbp = nsolc                            ! previous proc in vpar
  ighost_vparbn = ighost_vparbp + ghost_size_vpar  ! next proc in vpar
  ighost_mubp   = ighost_vparbn + ghost_size_vpar  ! previous proc in mu
  ighost_mubn   = ighost_mubp   + ghost_size_mu    ! next pro in mu
  ighost_vparbp_mubp = ighost_mubn + ghost_size_mu ! prev mu, prev vpar
  ighost_vparbn_mubp = ighost_vparbp_mubp + ghost_size_vpar_mu
  ighost_vparbp_mubn = ighost_vparbn_mubp + ghost_size_vpar_mu 
  ighost_vparbn_mubn = ighost_vparbp_mubn + ghost_size_vpar_mu
  ighost_vparbp_sbp  = ighost_vparbn_mubn + ghost_size_vpar_mu
  ighost_vparbn_sbp  = ighost_vparbp_sbp  + ghost_size_vpar_s
  ighost_vparbp_sbn  = ighost_vparbn_sbp  + ghost_size_vpar_s
  ighost_vparbn_sbn  = ighost_vparbp_sbn  + ghost_size_vpar_s
  ighost_sbp         = ighost_vparbn_sbn  + ghost_size_vpar_s
  ighost_sbn         = ighost_sbp         + ghost_size_s
  
  
  msolc = ighost_sbn + ghost_size_s
  ! Initialise the index function. msolc is the maximum value
  call index_init(lsendrecv_mu,lindex_vpar_last,parallel_s,msolc)

  !
  ! (6) Register all the offsets with the index function. This allows the
  !     index function to relate call patterns with offsets.
  !
  
  ! The main part of fdisi
  call register_offset(imod=0,ix=0,is=0,imu=0,ivpar=0,isp=0,ioffset=0)
  
  ! The fields, together with the tags that are used to reference them.
  ! The momentum conserving `field' also has a number of local species
  ! associated with it.
  call register_offset(ioffset=n_phi,ifield=iphi)
  call register_offset(ioffset=n_apar,ifield=iapar) 
  call register_offset(ioffset=n_bpar,ifield=ibpar) 
  call register_offset(ioffset=n_mom_conserve,ifield=i_mom,nsp=nsp)

  ! The ghost cells for fdisi
  call register_offset(ivpar=-1,ioffset=ighost_vparbp) 
  call register_offset(ivpar=+1,ioffset=ighost_vparbn)
  call register_offset(imu=-1,ioffset=ighost_mubp)
  call register_offset(imu=+1,ioffset=ighost_mubn)
  call register_offset(imu=-1,ivpar=-1,ioffset=ighost_vparbp_mubp)
  call register_offset(imu=-1,ivpar=+1,ioffset=ighost_vparbn_mubp)
  call register_offset(imu=+1,ivpar=-1,ioffset=ighost_vparbp_mubn)
  call register_offset(imu=+1,ivpar=+1,ioffset=ighost_vparbn_mubn)
  call register_offset(is=-1,ivpar=-1,ioffset=ighost_vparbp_sbp)
  call register_offset(is=-1,ivpar=+1,ioffset=ighost_vparbn_sbp)
  call register_offset(is=+1,ivpar=-1,ioffset=ighost_vparbp_sbn)
  call register_offset(is=+1,ivpar=+1,ioffset=ighost_vparbn_sbn)
  call register_offset(is=-1,ioffset=ighost_sbp)
  call register_offset(is=+1,ioffset=ighost_sbn)
  
  ! the ghost cells for the fields
  call register_offset(is=-1,ioffset=ighost_sbp+ioffset_phi,ifield=iphi)
  call register_offset(is=-1,ioffset=ighost_sbp+ioffset_apar,ifield=iapar)
  call register_offset(is=-1,ioffset=ighost_sbp+ioffset_bpar,ifield=ibpar)
  call register_offset(is=+1,ioffset=ighost_sbn+ioffset_phi,ifield=iphi)
  call register_offset(is=+1,ioffset=ighost_sbn+ioffset_apar,ifield=iapar)
  call register_offset(is=+1,ioffset=ighost_sbn+ioffset_bpar,ifield=ibpar)
  
  !
  ! (7) set up ghost points in the index function
  !
  
  call index_set_ghostpoints(gp_s=ghost_points_s,gp_mu=ghost_points_mu,&
      & gp_vpar=ghost_points_vpar,gp_vpar_mu=ghost_points_vpar_mu,     &
      & gp_vpar_s=ghost_points_vpar_s)
  
  !
  ! (8) For each set of points to be sent to adjacent processor ghost zones,
  !     create a datatype to aid communication ( i.e. in sending part of fdisi
  !     to the offset point on the receiving processor).
  !
  
  call buffer_pointer_setup
  
  ! Generate a new datatype TYPE_FDISI_IO for a mapping between the (variable)
  ! memory layout and the (presently fixed) restart file layout.
  call create_fdisi_io_type()

  
  
  ! Starting point for the reference of the fields  
  ! warning this assume a certain structure in which the fields are 
  ! stored in the solution. 
  nphi = indx(1,1,1,iphi)

  ! we now allocate everything else required in this module
  call dist_allocate(1)

end subroutine dist_init

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> create arrays for copying parts of the solution into communication buffers
!-----------------------------------------------------------------------------
subroutine buffer_pointer_setup
!-----------------------------------------------------------------------------

  use general, only : gkw_abort
  use control, only : nlapar, nlbpar
  use grid,    only : nvpar, ns, nsp, nx, nmod, nmu
  use index_function, only : index_invert_, index_reorder, indx_ 
  integer, allocatable :: bpi(:),bni(:)
  integer :: i, d1, k, d2, d3, d4,ii,ierr

  integer, dimension(6) :: starts,ends
  integer :: l,m,n,j
  integer :: i_s, i_sp, i_mu, i_vpar, i_x, i_mod

  if (ghost_size_s > 0) then
    ! allocate the bpi, bni
      allocate(bpi(ghost_size_s),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bpi')
      allocate(bni(ghost_size_s),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bni')
  ii=0
  call index_invert_(starts,ends,gps_next=ghost_points_s)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  call index_invert_(starts,ends,gps_next=ghost_points_s,field=iphi)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx(i_mod,i_x,i_s,iphi)

            end do
          end do
        end do
      end do
    end do
  end do
  if (nlapar) then
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
             ii=ii+1
             bni(ii) = indx(i_mod,i_x,i_s,iapar)
            end do
          end do
        end do
      end do
    end do
  end do
  end if
  if (nlbpar) then
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
             ii=ii+1
             bni(ii) = indx(i_mod,i_x,i_s,ibpar)
            end do
          end do
        end do
      end do
    end do
  end do
  end if
    if (ii /= ghost_size_s) call gkw_abort('bad num of gps 1')
  ii=0
  call index_invert_(starts,ends,gps_prev=ghost_points_s)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  call index_invert_(starts,ends,gps_prev=ghost_points_s,field=iphi)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx(i_mod,i_x,i_s,iphi)

            end do
          end do
        end do
      end do
    end do
  end do
  if (nlapar) then
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
             ii=ii+1
             bpi(ii) = indx(i_mod,i_x,i_s,iapar)
            end do
          end do
        end do
      end do
    end do
  end do
  end if
  if (nlbpar) then
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
             ii=ii+1
             bpi(ii) = indx(i_mod,i_x,i_s,ibpar)
            end do
          end do
        end do
      end do
    end do
  end do
  end if
    if (ii /= ghost_size_s) call gkw_abort('bad num of gps 2')
    call get_reordered_ghost_type(ghost_size_s,bni,TYPE_NEXT_S)
    call get_reordered_ghost_type(ghost_size_s,bpi,TYPE_PREV_S)
    if (allocated(bpi)) deallocate(bpi)
    if (allocated(bni)) deallocate(bni)
  end if
  if (ghost_size_vpar > 0) then
      allocate(bpi(ghost_size_vpar),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bpi')
      allocate(bni(ghost_size_vpar),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bni')
  ii=0
  call index_invert_(starts,ends,gpvpar_next=ghost_points_vpar)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
    if (ii /= ghost_size_vpar) call gkw_abort('bad num of gpvpar 1')
  ii=0
  call index_invert_(starts,ends,gpvpar_prev=ghost_points_vpar)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
    if (ii /= ghost_size_vpar) call gkw_abort('bad num of gpvpar 2')
    call get_reordered_ghost_type(ghost_size_vpar,bni,TYPE_NEXT_VPAR)
    call get_reordered_ghost_type(ghost_size_vpar,bpi,TYPE_PREV_VPAR)
    if (allocated(bpi)) deallocate(bpi)
    if (allocated(bni)) deallocate(bni)

    
  end if
  if (ghost_size_mu > 0) then
      allocate(bpi(ghost_size_mu),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bpi')
      allocate(bni(ghost_size_mu),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bni')
  ii=0
  call index_invert_(starts,ends,gpmu_next=ghost_points_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  ii=0
  call index_invert_(starts,ends,gpmu_prev=ghost_points_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
    call get_reordered_ghost_type(ghost_size_mu,bni,TYPE_NEXT_MU)
    call get_reordered_ghost_type(ghost_size_mu,bpi,TYPE_PREV_MU)
    if (allocated(bpi)) deallocate(bpi)
    if (allocated(bni)) deallocate(bni)

    
  end if

  ! VPAR-MU N.B. there are 4, FOUR(!) DATATYPES HERE!
  ! we sent 1 point to each diagonal processor
  if (ghost_size_vpar_mu > 0) then
      allocate(bpi(ghost_size_vpar_mu),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bpi')
      allocate(bni(ghost_size_vpar_mu),stat=ierr)
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bni')
  ii=0
  ! mu next, vpar next
  call index_invert_(starts,ends,gpmu_next=ghost_points_vpar_mu,gpvpar_next=ghost_points_vpar_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  ii=0
  ! mu prev, vpar prev
  call index_invert_(starts,ends,gpmu_prev=ghost_points_vpar_mu,gpvpar_prev=ghost_points_vpar_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
    call get_reordered_ghost_type(ghost_size_vpar_mu,bni,TYPE_NEXT_VPAR_NEXT_MU)
    call get_reordered_ghost_type(ghost_size_vpar_mu,bpi,TYPE_PREV_VPAR_PREV_MU)
  ! mu next, vpar prev
  ii=0
  call index_invert_(starts,ends,gpmu_next=ghost_points_vpar_mu,gpvpar_prev=ghost_points_vpar_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bni(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  ii=0
  ! mu prev, vpar next
  call index_invert_(starts,ends,gpmu_prev=ghost_points_vpar_mu,gpvpar_next=ghost_points_vpar_mu)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              ii=ii+1
              bpi(ii) = indx_(i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
            end do
          end do
        end do
      end do
    end do
  end do
  call get_reordered_ghost_type(ghost_size_vpar_mu,bni,TYPE_PREV_VPAR_NEXT_MU)
  call get_reordered_ghost_type(ghost_size_vpar_mu,bpi,TYPE_NEXT_VPAR_PREV_MU)

  
    if (allocated(bpi)) deallocate(bpi)
    if (allocated(bni)) deallocate(bni)

    
  end if
end subroutine buffer_pointer_setup

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine copies the potential from the distribution into the the phic
!> array. If nlphi = .false. phic is set to zero 
!-----------------------------------------------------------------------------
subroutine get_phi(fdis,phic)
!-----------------------------------------------------------------------------

  use general, only : gkw_abort
  use control, only : nlphi
  use grid,    only : nmod, nx, ns 

  complex, dimension(nsolc),     intent(in)    :: fdis
  complex, dimension(nmod,nx,ns),intent(inout) :: phic

! local parameters
  integer :: ix, i, imod 

  if (first_call) then
    call gkw_abort('get_phi: you can not call this before dist_init')
  endif


    if (nlphi) then 

      ! copy phi 
      do imod = 1, nmod 
        do ix = 1, nx
          do i = 1, ns
            phic(imod,ix,i) = fdis(indx(imod,ix,i,iphi))
          end do 
        end do 
      end do 
    
    else 
  
      ! phi is not solved for and therefore set to zero 
      do imod = 1, nmod 
        do ix = 1, nx 
          do i = 1, ns 
            phic(imod,ix,i) = (0.E0,0.E0) 
          end do 
        end do 
      end do 
    
    endif 
      

end subroutine get_phi

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine copies the vector potential from the distribution into the
!> aparc array. If nlapar = .false. then the vector potential is set to zero.
!-----------------------------------------------------------------------------
subroutine get_apar(fdis,aparc)
!-----------------------------------------------------------------------------

    use general, only : gkw_abort
    use control, only : nlapar
    use grid,    only : nmod, nx, ns 

  complex, intent(in)    :: fdis(nsolc)
  complex, intent(inout) :: aparc(nmod,nx,ns)

! local parameters
  integer :: ix, i, imod 
 
  if (first_call) then
    call gkw_abort('get_apar: you can not call this before dist_init')
  endif


    if (nlapar) then

      ! copy apar
      do imod = 1, nmod 
        do ix = 1, nx
          do i = 1, ns
            aparc(imod,ix,i) = fdis(indx(imod,ix,i,iapar))
          end do 
        end do 
      end do 
    
    else 
  
      ! set apar to zero
      do imod = 1, nmod 
        do ix = 1, nx 
          do i = 1, ns 
            aparc(imod,ix,i) = (0.E0,0.E0) 
          end do 
        end do 
      end do 
   
    endif 
      

end subroutine get_apar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine copies the parallel magnetic field perturbation from the 
!> distribution into the bparc array. If nlbpar = .false. then the parallel 
!> magnetic field perturbation is set to zero.
!-----------------------------------------------------------------------------
subroutine get_bpar(fdis,bparc)
!-----------------------------------------------------------------------------

    use general, only : gkw_abort
    use control, only : nlbpar
    use grid,    only : nmod, nx, ns 

  complex, intent(in)    :: fdis(nsolc)
  complex, intent(inout) :: bparc(nmod,nx,ns)

! local parameters
  integer :: ix, i, imod 
 
  if (first_call) then
    call gkw_abort('get_bpar: you can not call this before dist_init')
  endif


    if (nlbpar) then

      ! copy bpar
      do imod = 1, nmod 
        do ix = 1, nx
          do i = 1, ns
            bparc(imod,ix,i) = fdis(indx(imod,ix,i,ibpar))
          end do 
        end do 
      end do 
    
    else 
  
      ! set bpar to some uncommon value
      do imod = 1, nmod 
        do ix = 1, nx 
          do i = 1, ns 
            bparc(imod,ix,i) = (25.,35.) !(0.E0,0.E0) 
          end do 
        end do 
      end do 
   
    endif 
      

end subroutine get_bpar

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
!> Function that should do all the indexing for the the distribution function.
!> It is to be used everywhere in the code except for the optimised routines.
!
!-----------------------------------------------------------------------------
function indx_main(imod, ix, i_in, j_in, k_in, is)
!-----------------------------------------------------------------------------

  use index_function, only: indx_
  use control, only : vp_trap
  use grid,    only : ns, nx, nmod, nvpar, nmu, nsp, parallel_vpar,          &
                      & parallel_s, lsendrecv_mu

  integer :: indx_main

  integer, intent(in) :: ix, i_in, j_in, k_in, imod, is
  integer :: i, k, j, indx_offset,n_s,n_vpar,n_mu

  
  if (first_call) then
!    call gkw_abort('indx_main: you can not call this before dist_init')
  endif

  k = k_in
  i = i_in
  j = j_in
  indx_main = indx_(imod,ix,i,j,k,is)
  return

end function indx_main 

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
!>     function that should do all the indexing for the 
!>     potential, apar + anything else
!>     it is to be used everywhere in the code except for 
!>     the optimised routines. 
!
!-----------------------------------------------------------------------------
function indx_other(imod,ix,i_in,switch,species_number)
!-----------------------------------------------------------------------------

  use index_function, only : indx_
  use control, only : nlphi, nlapar, nlbpar, collisions
  use grid,    only : ns, nx, nmod, nvpar, nmu, nsp, parallel_s, lsendrecv_mu

  integer :: indx_other
  
  integer, intent(in) :: ix, i_in, imod, switch
  integer, optional, intent(in) :: species_number
  integer :: ifield_ref, n_s, i
  logical :: n_s_modified

  if (present(species_number)) then
    indx_other=indx_(switch,imod,ix,i_in,species_number)
  else
    indx_other=indx_(switch,imod,ix,i_in)
  end if    
  return
  
end function indx_other 

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine allocates all the arrays connected with dist 
!> requires 1 call for the arrays used in parallel_plans,
!> then called with "1" to allocate and "-1" to deallocate
!-----------------------------------------------------------------------------
subroutine dist_allocate(i)
!-----------------------------------------------------------------------------

  use general, only : gkw_abort
  use grid,    only : ns, nmu, nvpar, nx, nmod, nsp
  use control, only : vp_trap, lcalc_freq

  integer, intent(in) :: i 

  !  1 => allocate, 
  ! -1 => deallocate,

  !> integer for error status
  integer :: ierr

  ! initialize ierr
  ierr= 0

  if (i .eq. 1) then

    !
    ! allocation
    !

    ! allocate the array that contains the Maxwell 
    allocate(fmaxwl(ns,nmu,nvpar),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate fmaxwl')
  
    ! allocate the array that contains the alpha particle slowing down
    ! distribution
    allocate(falpha(ns,nmu,nvpar),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate falpha')
 
    ! allocate the array that contains phi 
    allocate(phi(nmod,nx,ns),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate phi')
  
    ! allocate the array that contains apar 
    allocate(apar(nmod,nx,ns),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate apar')
  
    ! allocate the array that contains bpar 
    allocate(bpar(nmod,nx,ns),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate bpar')
  
    ! allocate the distribution function 
    allocate(fdisi(nsolc),stat=ierr)
    if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate fdisi')

    ! allocate the arrays for the frequency calculation 
    if (lcalc_freq) then 
      allocate(phase(nmod,nx), stat = ierr) 
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate phase')
      allocate(last_phase(nmod,nx), stat = ierr) 
      if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate last_phase')
    endif
 
  else if (i .eq. -1) then
    
    !
    ! deallocation
    !
    
    if( allocated(fmaxwl) )  deallocate(fmaxwl)
    if( allocated(falpha) )  deallocate(falpha)
    if( allocated(phi) )     deallocate(phi)
    if( allocated(apar) )    deallocate(apar)
    if( allocated(bpar) )    deallocate(bpar)
    if( allocated(fdisi) )   deallocate(fdisi)
    if( allocated(phase) )   deallocate(phase) 
    if( allocated(last_phase) )   deallocate(last_phase) 

  else
    
    ! stop on bad input

    call gkw_abort('dist_allocate: called with i /= 1 or -1')

  endif
  
end subroutine dist_allocate

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Creates a new datatype for for communicating ghost points. The input is
!> an array of offsets `ghost_offset', and its length, `ghost_size'. The old
!> type is assumed to be complex; the returned type is `type_new'. 
!> `type_new' will be either a indexed block or a contiguous block of complex.
!-----------------------------------------------------------------------------

subroutine get_reordered_ghost_type(ghost_size,ghost_offset_in,type_new)

  integer, intent(in) :: ghost_size
  integer, intent(in), dimension(ghost_size) :: ghost_offset_in
  integer, allocatable, dimension(:) :: ghost_offset
  integer, intent(out) :: type_new

  integer :: type_old, i, u_count, blocklen, ierr

  type_old = MPICOMPLEX_X
  u_count  = 0
  blocklen = 1
  
  allocate(ghost_offset(ghost_size),stat=ierr)
  ghost_offset(:)=ghost_offset_in(:)
  ! check for contiguous and reduce offset by 1 for MPI call
  do i=1, ghost_size
  
    ! we don't know the exact offset, so just test with adjacent values
    if (i > 1) then
      if (ghost_offset(i) == ghost_offset(i-1)) u_count = u_count + 1
    end if
  
    ghost_offset(i) = ghost_offset(i) - 1
    
  end do

  if (.not. u_count == ghost_size - 1) then
    if (lverbose) then
      write (*,*) '* creating indexed block datatype of length',ghost_size
    end if
#if defined(mpi)
    call MPI_TYPE_CREATE_INDEXED_BLOCK(ghost_size,blocklen,ghost_offset,type_old,type_new, &
        & ierr)
#endif
  else
    if (lverbose) then
      write (*,*) '* creating contiguous block datatype of length',ghost_size
    end if
#if defined(mpi)
    call MPI_TYPE_CONTIGUOUS(ghost_size,type_old,type_new,ierr)
#endif
  end if
  
#if defined(mpi)
  call MPI_TYPE_COMMIT(type_new,ierr)
#else
  ! not very good, but there are no obvious better choices
  type_new = type_old
#endif
  
  deallocate(ghost_offset)

end subroutine get_reordered_ghost_type

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module dist
