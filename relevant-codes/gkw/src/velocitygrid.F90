module velocitygrid
! SVN:$Id: velocitygrid.F90 1005 2009-07-02 16:12:03Z  $

use general, only : gkw_abort
use mpiinterface

implicit none

private

public :: velgrid_init, connect_vpar, velocity_bound

!
! publicly available variables and parameters
!

public :: intmu,vpgr,mugr,dvp,dmu,intvp,iblow,ibhig

real :: dvp
real :: dmu
  
  !> global rms parallel velocity
  real, public :: vpgr_rms
  !> global rms mu
  real, public :: mugr_rms

!> the grid in parallel velocity space, vpgr(ns,nmu,nvpar) 
real, allocatable :: vpgr(:,:,:)
!> the grid in mu space,  mugr(nmu)
real, allocatable :: mugr(:)


!> the grid for velocity space integration, intmu(nmu) 
real, allocatable :: intmu(:)
!> the grid for velocity space integration, intvp(ns,nmu,nvpar)
real, allocatable :: intvp(:,:,:)

! Arrays that determine at which position in the s-grid 
! a trapped particle bounces. (only used for vp_trap =1) 
integer, allocatable :: iblow(:), ibhig(:)

! number of extra spaces needed in the parallel velocity direction
integer :: ivpar_extra

contains

subroutine velgrid_init

  call velgrid_allocate(1)
  call dist_grid_setup
  call nonuni_vel_grid_setup
  call velgrid_stats

end subroutine velgrid_init



!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!-----------------------------------------------------------------------------
!> This routine allocates all the arrays connected with the velocity grid
!> requires 1 call for the arrays used in parallel_plans,
!> then called with "1" to allocate and "-1" to deallocate
!-----------------------------------------------------------------------------
  subroutine velgrid_allocate(i)
!-----------------------------------------------------------------------------

use grid,    only : ns, nmu, nvpar, nx, nmod, vpmax, mumax
use control, only : vp_trap, order_of_the_scheme

integer, intent(in) :: i 
!  1 => allocate, 
! -1 => deallocate,

! the integer for the error message
integer :: ierr
logical verbose

! initialize the error parameter 
ierr= 0

if (i .eq. 1) then

  ! allocate the theta grid array
  ! in some cases we need vpgr from regions we do not solve in
  if (order_of_the_scheme .eq. 'second_order') then
    ivpar_extra = 1
    allocate(vpgr(ns,nmu,-1:nvpar+1),stat=ierr)
  else if (order_of_the_scheme .eq. 'fourth_order') then 
    ivpar_extra = 2
    allocate(vpgr(ns,nmu,-2:nvpar+2),stat=ierr)
  else
    call gkw_abort('velgrid_allocate: ivpar_extra; bad case of scheme order')
    ivpar_extra = 0 
    allocate(vpgr(ns,nmu,nvpar),stat=ierr)
  endif

  if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate vpgr')

  ! allocate the velocity grid 
  allocate(mugr(0:nmu+1),stat=ierr)
  if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate mugr')
  
  ! allocate the array for velocity space integration 
  ! vparallel-direction 
  allocate(intvp(ns,nmu,nvpar),stat=ierr)
  if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate intvp')
  
  ! allocate the array for velocity space integration
  ! mu-direction 
  allocate(intmu(nmu),stat=ierr)
  if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate intmu')
   
  if (vp_trap.eq.1) then 
     allocate(iblow(nvpar), stat = ierr) 
     if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate iblow')
     allocate(ibhig(nvpar), stat = ierr) 
     if (ierr.ne.0) call gkw_abort('dist_allocate: Could not allocate ibhig')
  endif
  
else if (i .eq. -1) then

! deallocation

  if( allocated(mugr) )    deallocate(mugr)
  if( allocated(vpgr) )    deallocate(vpgr)
  if( allocated(intmu) )   deallocate(intmu)
  if( allocated(intvp) )   deallocate(intvp)

else

  call gkw_abort('dist_allocate: called with i /= 1 or -1')

endif


return
  
  end subroutine velgrid_allocate


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!-----------------------------------------------------------------------------
!> This routine calculates the grids in velocity space as well as the arrays
!> that are necessary for the integration over velocity space. The integral
!> over the distribution function should then be calculated as
!> B_N(i) sum_{jk} intmu(j) intvp(k) f(j,k)
!-----------------------------------------------------------------------------
  subroutine dist_grid_setup 
!-----------------------------------------------------------------------------

use grid, only : n_mu_grid, imupb, imupe, nvpar, ns, nmu, &
                  & n_vpar_grid, ivparpb, ivparpe, vpmax, mumax

! local parameters
integer :: i, j, k
real :: dmu, pi, vperp, dvperp 
logical :: equal_mu_spacing 

! set equal_mu_spacing 
equal_mu_spacing = .false. 

! set the constant PI
pi  = 4.E0*atan(1.E0)

if (equal_mu_spacing) then 

  ! mu grid calculations  

  ! Calculate the mu grid values. 
  dmu = mumax /real(n_mu_grid)
  do j = imupb-1, imupe+1
     mugr(j-imupb+1) = (j-0.5)*dmu
  end do 

  ! Calculate the help array for the mu integration 
  do j = imupb, imupe
    intmu(j-imupb+1) = 2.E0*pi*dmu
  end do 

else 

  do j = 0 , n_mu_grid+1 

    ! calculate vperp 
    dvperp = sqrt(2.E0*mumax)/real(n_mu_grid)
    vperp = (j-0.5)*dvperp 

    if (j .ge. imupb-1 .and. j .le. imupe+1) then

      mugr(j-imupb+1) = vperp**2 / 2.E0

      if (j .ge. imupb .and. j .le. imupe) then
        intmu(j-imupb+1) = pi*((vperp+0.5E0*dvperp)**2 - (vperp-0.5*dvperp)**2)
      endif

    endif 
  end do 

endif 
 
!write(*,*) 'delta mu', dmu
! the parallel velocity grid 

! Caclulate the parallel velocity grid values 
dvp = 2.0*vpmax / real(n_vpar_grid) 
do i = 1, ns 
  do j = 1, nmu
    do k = ivparpb-ivpar_extra, ivparpe+ivpar_extra
      vpgr(i,j,k-ivparpb+1) = - vpmax + (k-0.5E0)*dvp 
    end do 
  end do 
end do 

! Calculate the help array for the parallel velocity integration 
do i = 1, ns 
  do j = 1, nmu 
    do k = 1, nvpar 
      intvp(i,j,k) = dvp 
    end do 
  end do 
end do 
  
  end subroutine dist_grid_setup 

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine velocity_bound(kin,jin,kout,jout,ingrid)
!--------------------------------------------------------------------
! This routine deals with the boundary conditions in velocity space
!--------------------------------------------------------------------
use grid, only : nvpar, nmu 

integer, intent(in) :: kin, jin 
integer :: kout, k_in, jout 
logical :: ingrid, ingrid_vpar
k_in=kin
ingrid = .false.

if (k_in.lt.1) then
  call connect_vpar(k_in,ingrid_vpar)
  if (.not. ingrid_vpar) then
    ingrid = .false.
    return
  endif
endif 
if (k_in.gt.nvpar) then 
  call connect_vpar(k_in,ingrid_vpar)
  if (.not. ingrid_vpar) then
    ingrid = .false.
    return
  endif
endif 
if (jin.gt.nmu) then 
  ingrid = .false. 
  return 
endif 
if (jin.lt.1) then 
  jout = 1-jin
  if (jout.gt.nmu) stop 'problem with the boundary conditions in mu' 
  ingrid = .true. 
  kout=kin
  return 
endif 

kout = kin 
jout = jin 
ingrid = .true. 
return 

end subroutine velocity_bound 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine connect_vpar(k,ingrid)
!--------------------------------------------------------------------
! This routine determines if a point is in the parallel velocity grid
!--------------------------------------------------------------------
use control, only : vp_trap
use grid,    only : nvpar, parallel_vpar, lproc_vpar_lowerb,&
                  & lproc_vpar_upperb

integer, intent(in) :: k
logical, intent(out) :: ingrid

! the default - the other cases below must be .true.
ingrid = .false.

! always in the vpar grid in this case 
if (k .ge. 1 .and. k .le. nvpar) ingrid = .true.

! in parallel_vpar case (with vp_trap = 0), we are in the grid
! for  k < 1, unless it is the first processor in vpar and for 
! k > nvpar, unless we are the last processor in vpar (i.e. on
! the boundary) Note that we let CONTROL deal with there being
! too few points per processor, depending on the order of the 
! scheme, so we should not need to deal with specific k here.
if (parallel_vpar .and. vp_trap .eq. 0) then
  if (k .lt. 1     .and. (.not. lproc_vpar_lowerb)) ingrid = .true.
  if (k .gt. nvpar .and. (.not. lproc_vpar_upperb)) ingrid = .true.
  if (k .lt. 1-2 .or. k .gt. nvpar + 2) then
    call gkw_abort('connect_vpar: something is wrong')
    ingrid = .false.
  endif
  ingrid = .true.
endif

return

end subroutine connect_vpar

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine nonuni_vel_grid_setup
    !-------------------------------------------------------------------
    ! This routine sets up the velocity grid with arbitrary number of
    ! points within the trapping condition determined by the value
    ! of n_trapped. 
    ! Must be called after parallelize_geom
    !-------------------------------------------------------------------
    use control, only : vp_trap
    use grid,    only : nvpar, ns, nmu, n_trapped, nperiod, N_s_grid
    use geom,    only : bn, bmin, bmax
    use general, only : gkw_warn

    real b, vpmax, delv, vpm, vpl,vpr, vpgr2, dvp1,dvp2,fdum,vdum,blim
    integer i, j, k, ierr, ndum, idum, m, kref
    logical searching, verbose

    ! dummy array (low field side velocity) 
    real, allocatable :: vp0(:,:)
    real, allocatable :: vlftemp(:,:)
    real, allocatable :: dvtemp(:,:)

    ! dummy array (interval of the parallel velocity grid)
    real, allocatable :: dvp(:) 

    ! dummy array (selected points) 
    integer, allocatable :: ivps(:)

    ! dummy array (parallel velocity grid including passing) 
    real, allocatable :: vplfs(:)
    !When number of bounce points is large, this is the array of 
    !values that will be kept.
    real, allocatable :: pkeep(:)

    ! check whether the parallel velocity grid is to be set up with 
    ! the parallel velocity following the trapping condition 
    if (vp_trap.ne.1) return 

    ! Set the error parameter 
    ierr = 0 
    verbose = .false.

    ! The set up in this routine assume that nperiod = 1. 
    if (nperiod.ne.1) call gkw_abort('With vp_trap = 1, nperiod must&
         & be equal to 1')

    ! nvpar must be even 
    if (mod(nvpar,2).ne.0) call gkw_abort('With vp_trap = 1 nvpar must&
         & be even')        
    
    if ((N_s_grid/2 -1).lt.n_trapped) call gkw_abort('n_trapped is too big. &
         & The number of possible bounce points (ns/2-1) is smaller than  &
         & number of points in the trapped zone -> See manual')
    !The number of trapped points must be smaller than the number of cells 
    !in the parallel velocity direction
    
    if ((nvpar/2).lt.n_trapped) call gkw_abort('With vp_trap = 1 n_trapped must be &
         & smaller than the number of parallel grid points nvpar/2')
    
    if(((nvpar/2)-n_trapped).eq.1) call gkw_abort('You only have one point in the & 
         & passing region of the velocity domain -> More are necessary.')

    ! Set the maximum parallel velocity (low field side) 
    vpmax = 3.E0 
    blim = 1.0E-13
    write(*,*) 'The bottom limit is',blim
    ! caclulate the low field side velocity for the trapped region 
    ! Only the positive values of v_parallel are caculated. These 
    ! are stored in vp0 
    allocate(vp0(nmu,ns/2+1),stat = ierr)
    if (ierr.ne.0) call gkw_abort('could not allocate vp0 in vel_grid_&
         &setup') 

    do j = 1, nmu
       do k = 1, ns/2  
          ! Determine the magnetic field at the bounce point of the 
          ! trapped particle -> Bounce points are defined exactly half way
          ! between points in ns to make differencing easier..possible relaxation
          ! of this in the future
          if (k.lt.ns/2) then 
             b = 0.5E0*(bn(ns/2 + k) + bn(ns/2 + k + 1)) 
          else 
             b = bmax 
          endif

          ! the velocity on the low field side 
          vp0(j,k) = sqrt(2.E0*mugr(j)*(b - bmin))
          !write(*,*) j,k,vp0(j,k),mugr(j)

       end do

       ! the trapped passing boundary -> Which is never
       !used anyway.
       vp0(j,ns/2+1) = sqrt(2.E0*mugr(j)*(bmax - bmin))
    end do

    ! allocate the array for the storage of the v_parallel interval
    ! and the array that holds the selected points 
    allocate(dvp(ns/2), stat = ierr) 
    if (ierr.ne.0) call gkw_abort('Unable to allocate dvp in vel_grid&
         &setup') 
    allocate(ivps(ns/2), stat = ierr) 
    if (ierr.ne.0) call gkw_abort('Unable to allocate ivps in vel_grid&
         &setup')  
    allocate(vlftemp(nvpar/2,nmu), stat = ierr) !vlftemp is a temporary
    !array with the low field side velocities
    if (ierr.ne.0) call gkw_abort('Unable to allocate vlftemp in vel_grid&
         &setup')  
    allocate(dvtemp(nvpar-1,nmu), stat = ierr) !vlftemp is a temporary
    !array with the low field side velocities
    if (ierr.ne.0) call gkw_abort('Unable to allocate vlftemp in vel_grid&
         &setup')  
    allocate(pkeep(ns/2-1), stat = ierr)
    if (ierr.ne.0) call gkw_abort('Unable to allocate pkeep in vel_grid&
         &setup')  
    call gkw_warn('At the moment doesnt work with odd numbers of ns')
    write(*,*)'!!!! The number of bounce points is ns/2 -1 when grid is even !!!!'
    idum = 0
    do i=1,(ns/2-1)
       pkeep(i) = 1
       idum = idum +1
    end do
    write(*,*)'Initially there are', idum, 'points kept. N_trapped is',n_trapped
    ndum = (ns/2-1)-n_trapped
    write(*,*)ndum, 'Points must be neglected'       
    idum = 0
    !n_trapped points must now be selected for within the trapped region

    do while(idum.lt.ndum)
       do k = 2, ns/2-2
          
          searching = .true. 
          m = k - 1 
          do while(searching)
             if (m.eq.0) then 
                vpl = 0.E0 
                searching = .false.
             else 
                if (pkeep(m).eq.1) then 
                   vpl = vp0(1,m) 
                   searching = .false. 
                else 
                   m = m - 1 
                endif
             endif
          end do
          
          searching = .true. 
          m = k + 1 
          do while (searching) 
             if (m.eq.ns/2+1) then 
                vpm = vp0(1,ns/2+1) 
                searching = .false. 
             else
                if (pkeep(m).eq.1) then 
                   vpm = vp0(1,m) 
                   searching = .false. 
                else 
                   m = m + 1 
                endif
             endif
          end do
          
          dvp(k) = vpm - vpl 
          if(verbose)then
             write(*,*)dvp(k),pkeep(k)
          end if
       end do
       !The point with the smallest interval is found and removed to even
       !out the resolution, and make sure n_trapped particles are kept.

       vpm=1
       do j = 2, ns/2-2  
          if (pkeep(j).eq. 1) then 
             if (dvp(j).lt.vpm) then 
                kref = j 
                vpm = dvp(j) 
             endif
          endif
       end do
       ! eliminate the minimum point 
       pkeep(kref) = 0 

       !Count the number of points that have been removed 
       idum = idum + 1  
       if(verbose)then
          write(*,*)'idum = ',idum
       end if
    end do
    
    if(verbose)then
       do i=1,ns/2-1
          write(*,*)'Poiint along ns', i, 'Is it kept?' , pkeep(i)
       end do
    end if
   
    !Build the grid at the low field point -> This is the reference for
    !all other points around the Torus, vpar0
    do j = 1,nmu
       
       !The points inside the trapped region are determined by the velocity
       !of bounce points calculated earlier 
       idum = 0       
       do i=1,ns/2-1
          if(pkeep(i).eq.1) then
             idum=idum+1
             vlftemp(idum,j) = vp0(j,i)
            ! write(*,*)'Point added'
          end if
       end do
       if(idum.ne.n_trapped) call gkw_abort('Something problem')
       vlftemp(n_trapped+1,j) = 2.E0*vp0(j,ns/2+1)- vlftemp(n_trapped,j) 
       !The first point outside the trapped region is equidistant from the
       !boundary as the first one inside
       
       dvp2 = ((vpmax - vlftemp(n_trapped+1,j))/(nvpar/2-n_trapped-1))
       !All other points outside this are evenly spaced
       write(*,*)'Velocity element outside trapped region = ',dvp2
       if(dvp2.lt.0)then
          call gkw_abort('Need more grid cells in the velocity grid outside &
               & the trapped region')
       end if
       
       if((nvpar/2).gt.n_trapped)then
          do i = n_trapped+2,nvpar/2
             vlftemp(i,j) = vlftemp(i-1,j) + dvp2
          end do
       end if
    end do
    
    !Following commmented section prints the grid velocity grid points
    !to a file.
    open(18,file = 'velgridtemp') !The grid at the high field point
    do j = 1, nmu 
       do k =  1, nvpar/2 
          write(18,*) sqrt(mugr(j)),vlftemp(k,j),vp0(j,ns/2)   
       end do
    end do

    deallocate(pkeep)

    !Section simultaneously calculates the velocity array along the field line
    !and symmetrises the array 
    do j = 1,nmu
       do i = 1, ns 
          do k = 1,nvpar/2 
             vpgr2 = 2.E0*mugr(j)*(bmin - bn(i)) + vlftemp(k,j)**2 
             if (vpgr2.gt.(0.E0)) then 
                vpgr(i,j,k+nvpar/2) = sqrt(vpgr2) 
                vpgr(i,j,nvpar/2-k+1) = -sqrt(vpgr2) 
                !vpgr(i,j,k+nvpar/2) = abs(vlftemp(k,j)) 
                !vpgr(i,j,nvpar/2-k+1) = - abs(vlftemp(k,j))
             else 
                vpgr(i,j,k+nvpar/2) = 0.E0
                vpgr(i,j,nvpar/2-k+1) = 0.E0 
             endif
          end do
       end do
    end do
    deallocate(vlftemp)

    ! build the array for the velocity space integration 
    !It is implemented so that it ignores any cells set
    !to zero, finds the next non zero cell and calculates
    !interval from that.  Symmetrised on the fly as well.
    do i = 1, ns
       do j = 1, nmu 
          do k = 1, nvpar/2 
             vpl = 0.E0
             vpr = 0.E0
             if (vpgr(i,j,k).eq.(0.E0)) then 
                intvp(i,j,k) = 0.E0 
                intvp(i,j,nvpar-k+1)=intvp(i,j,k)                 
             else 
                if (k.eq.1) then 
                   vpl =  vpmax - abs(vpgr(i,j,k))
                   kref = k+1
                   searching = .true.
                   do while(searching)
                      if(abs(vpgr(i,j,kref)).lt.blim) then
                         kref = kref+1
                      else
                         searching = .false.
                      end if
                   end do
                   vpr = 0.5*abs((vpgr(i,j,kref)-(vpgr(i,j,1))))
                   intvp(i,j,k) = vpr + vpl
                   intvp(i,j,nvpar-k+1)=intvp(i,j,k)        
                else if (k.eq.(nvpar/2)) then                   
                   vpr = abs(vpgr(i,j,nvpar/2))
                   kref=k-1
                   searching = .true.
                   do while(searching)
                      if(abs(vpgr(i,j,kref)).lt.blim)then
                         kref=kref-1
                      else
                         searching= .false.
                      end if
                   end do
                   vpl = 0.5*abs((vpgr(i,j,nvpar/2))-(vpgr(i,j,kref)))
                   intvp(i,j,nvpar/2) = vpr+vpl
                   intvp(i,j,nvpar-k+1)=intvp(i,j,k)    
                        
                else
                   kref = k+1
                   searching = .true.
                   do while(searching)
                      if(abs(vpgr(i,j,kref)).lt.blim)then
                         kref=kref+1
                      else
                         searching = .false.
                      end if
                   end do
                   vpr = 0.5*abs((vpgr(i,j,kref)-(vpgr(i,j,k))))            
                   kref=k-1
                   searching = .true.
                   do while(searching)
                      if(abs(vpgr(i,j,kref)).lt.blim)then
                         kref=kref-1
                      else
                         searching= .false.
                      end if
                   end do
                   vpl = 0.5*abs((vpgr(i,j,k))- (vpgr(i,j,kref)))
                   intvp(i,j,k) = vpr+vpl
                   intvp(i,j,nvpar-k+1)=intvp(i,j,k)
                end if

             endif
          end do
       end do
    end do

    !Following commmented section prints the grid velocity grid points
    !to a file.
    open(18,file = 'velgridtemp1') !The grid at the high field point
    do j = 1, nmu 
       do k =  1, nvpar 
          write(18,*) sqrt(mugr(j)),vpgr(1,j,k),vp0(j,1)   
       end do
    end do
    open(18,file = 'velgridtemp2') !The grid at the high field point
    do j = 1, nmu 
       do k =  1, nvpar 
          write(18,*) sqrt(mugr(j)),vpgr(2,j,k),vp0(j,2)   
       end do
    end do
    open(18,file = 'velgridtemp3') !The grid at the high field point
    do j = 1, nmu 
       do k =  1, nvpar
          write(18,*) sqrt(mugr(j)),vpgr(3,j,k),vp0(j,3)   
       end do
    end do
    open(18,file = 'velgridtemp4') !The grid at the high field point
    do j = 1, nmu 
       do k =  1, nvpar 
          write(18,*) sqrt(mugr(j)),vpgr(4,j,k),vp0(j,4)   
       end do
    end do

    ! Finally fill the arrays that determine in which point in the 
    ! s-grid the trapped particles bounce 
    do k = 1, nvpar/2 - n_trapped  
       iblow(k) = 0 
       ibhig(k) = 0 
    end do
    do k = nvpar/2 + n_trapped + 1, nvpar
       iblow(k) = 0
       ibhig(k) = 0
    end do
    do k = nvpar/2 - n_trapped + 1, nvpar/2 + n_trapped 
       !write(*,*) k
       iblow(k) = 1 
       m = 1 
       do while ((vpgr(m,1,k).eq.0).and.(m.le.ns))
          m = m + 1 
          iblow(k) = m 
       end do
       if (m.gt.ns/2) call gkw_abort('vel_grid_setup: can not find iblow')
       ibhig(k) = ns 
       m = ns 
       do while((vpgr(m,1,k).eq.0).and.(m.ge.1))
          m = m - 1 
          ibhig(k) = m 
       end do
       if (m.lt.ns/2) call gkw_abort('vel_grid_setup: can not find ibhig')
    end do

  end subroutine nonuni_vel_grid_setup


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> calculate the rms values of the velocity grid
!----------------------------------------------------------------------------
subroutine velgrid_stats

  use mpiinterface
  use grid, only : ns,nmu,nvpar

  integer :: i,j,k
  integer :: ierr
  real, dimension(2) :: vpmu, buf
  
  vpmu(1) = sum(vpgr(1:ns,1:nmu,1:nvpar)**2)/(1.*ns*nmu*nvpar*number_of_processors)
  vpmu(2) = sum(mugr(1:nmu)**2)/(1.*nmu*number_of_processors)

  call mpiallreduce_sum(vpmu,buf,2)

  vpgr_rms = sqrt(buf(1))
  mugr_rms = sqrt(buf(2))

end subroutine velgrid_stats

end module velocitygrid
