module collisionop
! SVN:$Id: collisionop.F90 1020 2009-07-02 19:56:45Z  $

use global,  only : lverbose
use grid,    only : lproc_mu_upperb, lproc_mu_lowerb, lproc_vpar_lowerb,       &
                  & lproc_vpar_upperb
use control, only : lcollisions => collisions
use general, only : gkw_abort
use specfun, only : erf => sf_erf
use mpiinterface
use mpicomms

implicit none

private

public :: collision_operator_setup, mom_conservation
public :: collisionop_read_nml, collisionop_write_nml, collisionop_bcast_nml
public :: collisionop_check_params, coll_mom_change_int, coll_mom_change_diag

interface collisionop_write_nml
   module procedure collisionop_read_nml
end interface

!> The reference major radius (only used in the collision operator)
real :: rref 

!> The reference temperature in units of keV (only used for the 
!> collision operator 
real :: tref 

!> The reference density in units 10^19 m^-3 (only used for the 
!> collision operator 
real :: nref 

!Switches for the various terms within the collision operator
!Pitch angle scattering
logical pitch_angle
!Energy scattering
logical en_scatter
!Friction
logical friction_coll
!Switch whether or not we desire momentum to be conserved in the collision
!operator.  Significant slow down due to integrations.
logical mom_conservation

!Mass conservation requires a slightly unphysical boundary condition
!the diffusion coefficients on the boundaries are set to zero for zero
!flux across the boundary.
!If false the boundaries are open and mass can flow out of the domain
!and therefore mass is no longer conserved.
!Mass conservation only works to machine precision when regular mu spacing
!is used
logical mass_conserve 

!Parameter from the collision namelist, set to true of user defined collision
!frequency is wanted
logical freq_override 

!User defined collision frequecy, which in our units should be fed in for a Z=1
!species at the reference thermal velocity
real coll_freq

!> The factor Gamma^a/b of the collision operator 
real, allocatable :: gammab(:,:)

! various globals needed here
real, allocatable, dimension(:) :: G_vthrat,G_mas,G_tmp,G_de,G_signz

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This subroutine reads (or writes) the reference value of the mass the
!> density and the temperature. These are needed only to calculate the
!> collision frequency. 
!----------------------------------------------------------------------------

subroutine collisionop_read_nml(lun,io_stat,lwrite)

  integer, intent(in)  :: lun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /collisions/ nref, tref, rref , pitch_angle, en_scatter,          &
                      & friction_coll, mom_conservation, mass_conserve,      &
                      & freq_override, coll_freq
                      

  io_stat = 0

  if (.not. lcollisions) then
    call gkw_abort('collisionop_read_nml: called with collisions = F')
  end if
  
  if (present(lwrite)) then
    if (.not. lwrite) then
    
      ! Default values..overidden by namelist read
      rref = 1.
      tref = 1.
      nref = 1.
      pitch_angle = .true.
      en_scatter = .true.
      friction_coll = .true.
      mom_conservation = .false.
      mass_conserve = .true.
      freq_override = .false.
      coll_freq =0.
      
      read(lun,NML=collisions,IOSTAT=io_stat) 
    else
      ! do nothing
    end if
  else
    write(lun,NML=collisions)
  end if 

end subroutine collisionop_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> bcast the collisionop namelist params
!----------------------------------------------------------------------------

subroutine collisionop_bcast_nml

  call mpibcast_real(rref, 1)
  call mpibcast_real(tref, 1)
  call mpibcast_real(nref, 1)
  call mpibcast_logical(pitch_angle, 1)
  call mpibcast_logical(en_scatter, 1)
  call mpibcast_logical(friction_coll, 1)
  call mpibcast_logical(mom_conservation, 1)
  call mpibcast_logical(mass_conserve, 1)
  call mpibcast_logical(freq_override, 1)
  call mpibcast_real(coll_freq,  1)

end subroutine collisionop_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> put any checks that can be done before memory allocation in here
!----------------------------------------------------------------------------

subroutine collisionop_check_params

if(freq_override) then
   if(lverbose)then
      !APS: change these to warns
      write(*,*)
      write(*,*)'User defined collision frequencies selected'
      write(*,*)'rref input ignored'
      write(*,*)'tref, nref still used in Couloumb logarithm' 
      write(*,*) 
   end if
   !Change values to be written back to input.out
   rref = 0.
   
 end if

end subroutine collisionop_check_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!Collisions initialisation
subroutine collision_operator_setup

use grid,    only : nsp,number_of_species,nvpar, nmu, nx, nmod, ns
use control, only : collisions,vp_trap

!Error parameter
integer ierr
ierr = 0

!All the collision operator setup is performed here 
allocate(gammab(number_of_species,number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate gammab in collisions'
endif
allocate(G_vthrat(number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate G_ in collisions'
endif
allocate(G_mas(number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate G_ in collisions'
endif
allocate(G_de(number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate G_ in collisions'
endif
allocate(G_tmp(number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate G_ in collisions'
endif
allocate(G_signz(number_of_species), stat = ierr) 
if (ierr.ne.0) then 
   stop 'Could not allocate G_ in collisions'
endif

! call the routine that initializes the factor for the collision operator
call collision_init

if(vp_trap.eq.0)then
   call collision_op_uniform 
   if(friction_coll)then
      call collision_friction_uniform
   end if
   if(mom_conservation)then
      call cons_momentum
   end if
else
   call gkw_abort('Error in Collision operator call')
end if

! APS: this can be deallocated in any case now, right?
if(.not. mom_conservation)then
   if (allocated(gammab)) deallocate(gammab)
end if

if (lverbose) then
   write(*,*)'Collision operator initialisation complete'
end if

return

end subroutine collision_operator_setup

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!
!> This subroutine caclulates Gamma(a,b) necessary for the collision 
!> operator. Gamma is defined as 
!>
!>  gammab = R_ref nb q_a^2 q_b^2 lambda^{a/b} /
!>           (4 pi epsilon_0^2 m_a vth_a^4) 
!> 
!> where R_ref is the reference major radius [in meters] 
!>       nb is the density of the species b
!>       q_a is the particle charge of species a 
!>       q_b is the charge of the particle of species b
!>       lambda is the Coulomb logarithm
!>       m_a is the particle mass (species a)  
!>       vth_a is the thermal velocity (species a) 
!
!--------------------------------------------------------------------
subroutine collision_init 

use grid, only : nsp, number_of_species
use dist,    only : iadia 
use components,   only :  adiabatic_electrons, de,mas,tmp,vthrat,signz


! integers for the loops 
integer i, j 

! the electron density 
real electron_density, electronmass, iigamma

!The reference gamma calculated from the input collision frequency
real ref_freq
 
logical :: ALL_PROCS

ALL_PROCS = .true.
! get the global species dependent quantities
call gather_array(G_mas,number_of_species,mas(1:nsp),nsp,COMM_SP_NE,ALL_PROCS)
call gather_array(G_de,number_of_species,de(1:nsp),nsp,COMM_SP_NE,ALL_PROCS)
call gather_array(G_tmp,number_of_species,tmp(1:nsp),nsp,COMM_SP_NE,ALL_PROCS)
call gather_array(G_vthrat,number_of_species,vthrat(1:nsp),nsp,COMM_SP_NE,ALL_PROCS)
call gather_array(G_signz,number_of_species,signz(1:nsp),nsp,COMM_SP_NE,ALL_PROCS)


! determine the electron density in units of 10^19 
if (adiabatic_electrons) then 
  electron_density = de(nsp+iadia) * nref 
else 
  ! find the electron species
  do j = 1, number_of_species
    if (G_signz(j) < 0) then 
      electron_density = G_de(j) * nref 
    endif 
  end do 
endif 

! calculate the Coulomb logarithm. Note the calculation is stored 
! in gammab, which is later transformed in the pre-factor gamma(a,b)
! of the collision operator 
do i = 1, number_of_species
   do j = 1, number_of_species
      ! check for electrons 
      if (G_signz(i) < 0.) then
         if (G_signz(j) < 0.) then 
            ! electron electron collisions 
            gammab(i,j) = 14.9E0 - 0.5E0*log(0.1E0*electron_density) +  &
                 & log(G_tmp(i)*tref)
         else 
            ! electron ion collisions 
            gammab(i,j) = 15.2E0 - 0.5E0*log(0.1E0*electron_density) +  &
                 & log(G_tmp(i)*tref) 
         endif
      else  
         if (G_signz(j) < 0.) then
            ! ion electron collisions  
            gammab(i,j) = 15.2E0 - 0.5E0*log(0.1E0*electron_density) +  &
                 & log(G_tmp(i)*tref) 
         else
            ! ion -ion collisions 
            gammab(i,j) = 17.3E0 - 0.5E0*log(0.1E0*electron_density) +  &
                 & 1.5*(0.5E0*(G_tmp(i)+G_tmp(j))*tref)
         endif
      endif
   end do
end do

if(freq_override)then
   !The input collision frequency is assumed to be the electron ion scattering
   !freqency.  Therefore the background density is the ion density and the scattered
   !species mass is the electron mass.
      
   iigamma =  gammab(1,1)
   ref_freq = coll_freq
   if (lverbose) then
     write(*,*)'!Normalised collision frequencies! First index scattered species'
     write(*,*)'!Second index is the scattering species!'
   end if
   do i = 1,number_of_species
      do j=1,number_of_species
         !Input freqency (single charged ion-ion frequency) should be calculated from
         !nu = 6.5141x10^{-5}*Rref*logLii*nref/Tref^2 and fed in as ref_freq
         !This is the ion-ion frequency, then is rescaled according to the expression
         !below
         gammab(i,j) = G_signz(i)**2*G_signz(j)**2*ref_freq*G_de(j)*(gammab(i,j)/iigamma)/(1.E0*G_tmp(i)**2)
         if (lverbose) write(*,*)i,j,gammab(i,j)
      end do
   end do
else !not freq_override
   if (lverbose) then
     write(*,*)'!Normalised collision frequencies! First index scattered species'
     write(*,*)'!Second index is the scattering species!'
   end if
   do i = 1, number_of_species
      do j = 1, number_of_species
         !Alternatively if freq_override is false it is calculated as follows
         !which is kinda the same
         gammab(i,j) = 6.5141e-5*(rref*nref/(tref**2))*G_de(j)*G_signz(i)**2 &
              & *G_signz(j)**2*gammab(i,j)/(G_tmp(i)**2)
         if (lverbose) write(*,*)i,j,gammab(i,j)
      end do
   end do
end if !freq_override

return 
end subroutine collision_init 

!--------------------------------------------------------------------
!> This routine calculates the diffusion coefficient of pitch angle 
!> scattering. Note dthth is not exactly D_theta,theta one usually 
!> find in the literature. It is defined here as 
! 
!> dthth = sum_b (1/4) gamma^(a/b)_N [ (2-1/vtb**2) erf(vtb) + 
!         erf^prime(vtb) / vtb ] 
!
!--------------------------------------------------------------------
subroutine caldthth(vp,mubn,is,dthth)

use grid,    only : number_of_species
use components, only : vthrat

real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   dthth 

integer :: i 
real vn, vtb, dum

vn = sqrt(vp**2 + 2.E0*sqrt(mubn**2)) 

dthth = 0. 
! N.B. _is_ corresponds to the local species
do i = 1, number_of_species 
  vtb = vn*vthrat(is) / G_vthrat(i) 
  dum = (2.E0 - 1.E0/(vtb**2))*erf(vtb) + erfp(vtb)/vtb 
  dthth = dthth + gammab(gsn(is),i)*dum/(4.E0*vn) 
end do 

end subroutine caldthth 


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!> This routine calculates the diffusion coefficient of energy 
!> scattering. It is defined here as 
! 
!> dvv = sum_b (1/2) gamma^(a/b)_N [ (1/vtb**2) erf(vtb) - 
!         erf^prime(vtb) / vtb ] 
!
!--------------------------------------------------------------------
subroutine caldvv(vp,mubn,is,dvv)

use grid,   only : number_of_species
use components, only : vthrat

real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   dvv 

integer i 
real vn,  vtb, dum

vn = sqrt(vp**2 + 2.E0*mubn) 

dvv = 0. 
do i = 1, number_of_species 
  vtb = vn * vthrat(is) / G_vthrat(i) 
  dum = erf(vtb)/(vtb**2) - erfp(vtb)/vtb 
  dvv = dvv + gammab(gsn(is),i)*dum/(2.E0*vn) 
end do 

end subroutine caldvv

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!> This routine calculates the diffusion coefficient of energy 
!> scattering. It is defined here as 
! 
!> dvv = sum_b (1/2) gamma^(a/b)_N [ (1/vtb**2) erf(vtb) - 
!         erf^prime(vtb) / vtb ] 
!
!--------------------------------------------------------------------
subroutine selfcaldvv(vp,mubn,is,dvv)

real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   dvv 

integer i 
real vn,  vtb, dum

vn = sqrt(vp**2 + 2.E0*mubn) 

dvv = 0.  
dum = erf(vn)/(vn**2) - erfp(vn)/vn 
dvv = gammab(gsn(is),gsn(is))*dum/(2.E0*vn) 

end subroutine selfcaldvv


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!> This is a copy of the above caldthth function, but instead of sum
!> ming over all species only considers self collisions for the 
!> momentum conserving term
!--------------------------------------------------------------------
subroutine selfcaldthth(vp,mubn,is,dthth)
 
real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   dthth 

integer i 
real vn, vtb, dum 

vn = sqrt(vp**2 + 2.E0*sqrt(mubn**2)) 
dthth = 0.   
dum = (2.E0 - 1.E0/(vn**2))*erf(vn) + erfp(vn)/vn 
dthth = gammab(gsn(is),gsn(is))*dum/(4.E0*vn) 

end subroutine selfcaldthth 


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!!$!--------------------------------------------------------------------
!!$!> This routine calculates the friction coefficient. It is defined 
!!$!> here as 
!!$! 
!!$!> fv = -sum_b (ma/mb) gamma^(a/b)_N [ erf(vtb) - 
!!$!         vtb erf^prime(vtb)  ] 
!!$!
!!$!--------------------------------------------------------------------
subroutine calfv(vp,mubn,is,fv)

use grid,   only : number_of_species
use components, only : vthrat, mas

real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   fv,mrat

integer i 
real vn,  vtb, dum

!Need to calculate the mass ratio.

vn = sqrt(vp**2 + 2.E0*mubn) 

fv = 0. 
do i = 1, number_of_species
   mrat = mas(is)/G_mas(i)
   vtb = vn * vthrat(is)/G_vthrat(i)
   dum = erf(vtb) - erfp(vtb)*vtb 
   fv = fv + mrat*gammab(gsn(is),i)*dum/(vn**2)  
end do 

end subroutine calfv

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!!$!--------------------------------------------------------------------
!!$!> This routine calculates the friction coefficient. It is defined 
!!$!> here as 
!!$! 
!!$!> fv = -sum_b (ma/mb) gamma^(a/b)_N [ erf(vtb) - 
!!$!         vtb erf^prime(vtb)  ] 
!!$!
!!$!--------------------------------------------------------------------
subroutine selfcalfv(vp,mubn,is,fv)

real,  intent(in) :: vp, mubn 
integer, intent(in) :: is  
real   fv,mrat

integer i 
real vn,  vtb, dum

!Need to calculate the mass ratio.

vn = sqrt(vp**2 + 2.E0*mubn) 

fv = 0. 
dum = erf(vn) - erfp(vn)*vn 
fv = fv + gammab(gsn(is),gsn(is))*dum/(vn**2)  

end subroutine selfcalfv

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine collision_op_uniform
!----------------------------------------------------------------
!Collision operator like
!
!----------------------------------------------------------------
use control,    only : vp_trap
use grid,       only : nmod, nx, ns, nmu, nvpar, nsp, n_vpar_grid
use grid,       only : vpmax
use dist,       only : indx
use geom,       only : bn 
use components, only : vthrat
use matdat,     only : put_element 
use velocitygrid, only : mugr, vpgr, dvp, dmu, intvp, intmu

implicit none

integer imod, ix, i, j, k, is, idum
integer iih, jjh,jdum,kdum
logical ingrida, ingridb

!Values of diffusion coefficent in the 4 half grid points of interest
!and the point of consideration
!Temporary storage of the diffusion coefficients and the prefactors
real Dtemp
!Half point velocity values
real vphalf, vmhalf, muphalf, mumhalf, dmup, dmud
real fac,fad
complex mat_elem
!For both uniform and non-uniform grids

!The grid is uniform in the mu direction, therefore
!this is calculated only once. 
do is = 1,nsp   
   do imod = 1, nmod
      do ix = 1,nx
         do i = 1,ns
            

            do j = 1,nmu
               do k = 1,nvpar
                  
                  
                  !The point in the array we're interested in
                  iih = indx(imod,ix,i,j,k,is) 
             
                  vphalf = vpgr(i,j,k) + 0.5*dvp 
                  vmhalf = vpgr(i,j,k) - 0.5*dvp
                  
                  !The mu intervals above and below the point of
                  !interest
 
                  if(j.eq.1 .and. lproc_mu_lowerb)then
                     dmup = mugr(2)-mugr(1)
                     dmud = 2.E0*mugr(1)
                  else
                     dmup = mugr(j+1)-mugr(j)
                     dmud = mugr(j)-mugr(j-1)
                  end if
                  
                  !The half grid mu values
                  muphalf = mugr(j)+0.5*dmup
                  mumhalf = mugr(j)-0.5*dmud
                  
                  !Defined as the interval between the two half 
                  !points
                  dmu = muphalf - mumhalf
                  
                  !First is the dvpardvpar term
                  if(pitch_angle)then
                     if((k.eq.nvpar).and.mass_conserve.and.lproc_vpar_upperb)then
                        fac = 0.E0
                     else
                        call caldthth(vphalf,mugr(j)*bn(i),is,Dtemp)
                        fac = 2*vthrat(is)*mugr(j)*Dtemp*bn(i)       & 
                             & /(vphalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if
                 
                  if(en_scatter)then
                     if((k.eq.nvpar).and.mass_conserve.and.lproc_vpar_upperb)then
                        fad = 0.E0
                     else
                        call caldvv(vphalf,mugr(j)*bn(i),is,Dtemp)
                        fad = vthrat(is)*(vphalf**2)*Dtemp &
                             &/(vphalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if

                  jdum=j
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = (fac+fad)/(dvp*dvp)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -(fac+fad)/(dvp*dvp) 
                     call put_element(iih,jjh,mat_elem)
                  end if

                  if(pitch_angle)then
                     if((k.eq.1).and.mass_conserve.and.lproc_vpar_lowerb)then
                        fac = 0.E0
                     else
                        call caldthth(vmhalf,mugr(j)*bn(i),is,Dtemp)              
                        fac =  -2*vthrat(is)*mugr(j)*Dtemp*bn(i)     & 
                             & /(vmhalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if
                  
                  if(en_scatter)then
                     if((k.eq.1).and.mass_conserve.and.lproc_vpar_lowerb)then
                        fad = 0.E0
                     else
                        call caldvv(vmhalf,mugr(j)*bn(i),is,Dtemp)
                        fad = -vthrat(is)*(vmhalf**2)*Dtemp &
                             &/(vmhalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if
                  
                  jdum=j
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = (fac+fad)/(dvp*dvp)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -(fac+fad)/(dvp*dvp) 
                     call put_element(iih,jjh,mat_elem)
                  end if

                  !Second is the dmudmu term    
                  
                  if(pitch_angle)then
                     if((j.eq.nmu).and.mass_conserve.and.lproc_mu_upperb)then
                        fac = 0.E0
                     else
                        call caldthth(vpgr(i,j,k),muphalf*bn(i),is,Dtemp)    
                        fac = 2*vthrat(is)*muphalf*Dtemp*vpgr(i,j,k)**2     & 
                             & /((vpgr(i,j,k)**2 + 2*muphalf*bn(i))*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if

                  if(en_scatter)then
                     if((j.eq.nmu).and.mass_conserve.and.lproc_mu_upperb)then
                        fad = 0.E0
                     else
                        call caldvv(vpgr(i,j,k),muphalf*bn(i),is,Dtemp)
                        fad = 4*vthrat(is)*(muphalf**2)*Dtemp &
                             &/(vpgr(i,j,k)**2 + 2*muphalf*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if

                  jdum=j+1
                  kdum=k               
                  call vgridboundary(i,jdum,kdum,ingrida) 
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = (fac+fad)/(dmu*dmup)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -(fac+fad)/(dmu*dmup) 
                     call put_element(iih,jjh,mat_elem)
                  end if
                          
                  if(pitch_angle)then
                     call caldthth(vpgr(i,j,k),mumhalf*bn(i),is,Dtemp) 
                     fac =  -2*vthrat(is)*mumhalf*Dtemp*vpgr(i,j,k)**2     &                       
                          & /((vpgr(i,j,k)**2 + 2*mumhalf*bn(i))*bn(i))
                  else
                     fac = 0.E0
                  end if

                  if(en_scatter)then
                     call caldvv(vpgr(i,j,k),mumhalf*bn(i),is,Dtemp)
                     fad = -4*vthrat(is)*(mumhalf**2)*Dtemp &
                          &/(vpgr(i,j,k)**2 + 2*mumhalf*bn(i))
                  else
                     fad = 0.E0
                  end if

                  jdum=j
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then               
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = (fac+fad)/(dmu*dmud)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j-1
                  kdum=k   
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -(fac+fad)/(dmu*dmud) 
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  
                  !Cross terms
                  !Must interpolate (using Bilinear) the half/half grid
                  !points
                 
                  !Firstly dmudvpar
                  if(pitch_angle)then
                     if((j.eq.nmu).and.mass_conserve.and.lproc_mu_upperb)then
                        fac=0.E0
                     else
                        call caldthth(vpgr(i,j,k),muphalf*bn(i),is,Dtemp)
                        fac =   -2*vthrat(is)*muphalf*Dtemp*vpgr(i,j,k)     & 
                             & /(vpgr(i,j,k)**2 + 2*muphalf*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if

                  if(en_scatter)then
                     if((j.eq.nmu).and.mass_conserve.and.lproc_mu_upperb)then
                        fad=0.E0
                     else
                        call caldvv(vpgr(i,j,k),muphalf*bn(i),is,Dtemp)
                        fad = 2*vthrat(is)*vpgr(i,j,k)*muphalf*Dtemp  &
                             &/(vpgr(i,j,k)**2 + 2*muphalf*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if

                  jdum=j+1
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)                  
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  jdum=j+1
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)                     
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  if(pitch_angle)then
                     if((j.eq.1).and.mass_conserve.and.lproc_mu_lowerb)then
                        fac=0.E0
                     else
                        call caldthth(vpgr(i,j,k),mumhalf*bn(i),is,Dtemp) 
                        fac = 2*vthrat(is)*mumhalf*Dtemp*vpgr(i,j,k)     & 
                             & /(vpgr(i,j,k)**2 + 2*mumhalf*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if
                  
                  if(en_scatter)then
                     if((j.eq.1).and.mass_conserve.and.lproc_mu_lowerb)then
                        fad=0.E0
                     else
                        call caldvv(vpgr(i,j,k),mumhalf*bn(i),is,Dtemp)
                        fad =  -2*vthrat(is)*vpgr(i,j,k)*mumhalf*Dtemp  &
                             &/(vpgr(i,j,k)**2 + 2*mumhalf*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if 
                  
                  jdum=j
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j-1
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  jdum=j-1
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)  
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh =indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  !Then dvpardmu
                  if(pitch_angle)then
                     if((k.eq.nvpar).and.mass_conserve.and.lproc_vpar_upperb)then
                        fac = 0.E0
                     else
                        call caldthth(vphalf,mugr(j)*bn(i),is,Dtemp)
                        fac =  -2*vthrat(is)*mugr(j)*Dtemp*vphalf     & 
                             & /(vphalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fac = 0.E0
                  end if

                  if(en_scatter)then
                     if((k.eq.nvpar).and.mass_conserve.and.lproc_vpar_upperb)then
                        fad = 0.E0
                     else
                        call caldvv(vphalf,mugr(j)*bn(i),is,Dtemp)
                        fad = 2*vthrat(is)*mugr(j)*vphalf*Dtemp  &
                             & /(vphalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if
                  
                  jdum=j+1
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j+1
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  jdum=j-1
                  kdum=k+1
                  call vgridboundary(i,jdum,kdum,ingrida)                 
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j-1
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)       
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  if(pitch_angle)then
                     if((k.eq.1).and.mass_conserve.and.lproc_vpar_lowerb)then
                        fac = 0.E0
                     else
                        call caldthth(vmhalf,mugr(j)*bn(i),is,Dtemp)
                        fac =   2*vthrat(is)*mugr(j)*Dtemp*vmhalf     & 
                             & /((vmhalf**2 + 2*mugr(j)*bn(i)))
                     end if
                  else
                     fac = 0.E0
                  end if
                  
                  if(en_scatter)then 
                     if((k.eq.1).and.mass_conserve.and.lproc_vpar_lowerb)then
                        fad = 0.E0 
                     else
                        call caldvv(vmhalf,mugr(j)*bn(i),is,Dtemp)
                        fad = -2*vthrat(is)*mugr(j)*vmhalf*Dtemp  &
                             & /(vmhalf**2 + 2*mugr(j)*bn(i))
                     end if
                  else
                     fad = 0.E0
                  end if

                  jdum=j+1
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)       
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j+1
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)   
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  jdum=j-1
                  kdum=k
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j-1
                  kdum=k-1
                  call vgridboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.25*(fac+fad)/(dvp*dmu)
                     call put_element(iih,jjh,mat_elem)
                  end if
             
               end do
            end do
         end do
         
      end do
   end do
end do

return 

end subroutine collision_op_uniform

!----------------------------------------------------------------
!----------------------------------------------------------------
subroutine collision_friction_uniform
!----------------------------------------------------------------
!Collision operator like
!
!----------------------------------------------------------------
use grid,       only : nmod, nx, ns, nmu, nvpar, nsp, n_vpar_grid
use grid,       only : vpmax
use control,    only : vp_trap
use dist,       only : indx
use geom,       only : bn 
use components, only : vthrat
use matdat,     only : put_element 
use velocitygrid, only : mugr, vpgr, dvp, dmu

implicit none

integer imod, ix, i, j, k, is, idum
integer iih, jjh,jdum,kdum
logical ingrida, ingridb

!Values of diffusion coefficent in the 4 half grid points of interest
!and the point of consideration
!Temporary storage of the diffusion coefficients and the prefactors
real vphalf, vmhalf, muphalf, mumhalf
real Dtemp
!Half point velocity values
real fac, dmup, dmum
complex mat_elem
!For both uniform and non-uniform grids

!The grid is uniform in the mu direction, therefore
!this is calculated only once.

if (lverbose) then
   write(*,*) 'Friction term called'
end if

do imod = 1, nmod
   do ix = 1,nx
      do i = 1,ns
         do j = 1,nmu
            do k = 1,nvpar
               do is = 1,nsp   
           

                  !The point in the array we're interested in
                  iih = indx(imod,ix,i,j,k,is) 
             
                  vphalf = vpgr(i,j,k) + 0.5*dvp 
                  vmhalf = vpgr(i,j,k) - 0.5*dvp
                  
                  !The mu intervals above and below the point of
                  !interest
 
                  if(j.eq.1 .and. lproc_mu_lowerb)then
                     dmup = mugr(2)-mugr(1)
                     dmum = 2.E0*mugr(1)
                  else
                     dmup = mugr(j+1)-mugr(j)
                     dmum = mugr(j)-mugr(j-1)
                  end if
                  
                  !The half grid mu values
                  muphalf = mugr(j)+0.5*dmup
                  mumhalf = mugr(j)-0.5*dmum
                  
                  !Defined as the interval between the two half 
                  !points
                  dmu = muphalf - mumhalf
                  
                  !d/dvpar friction term
                  if((k.eq.nvpar).and.mass_conserve.and.lproc_vpar_upperb)then
                     fac = 0.E0
                  else
                     call calfv(vphalf,mugr(j)*bn(i),is,Dtemp) 
                     fac = 1.E0*vthrat(is)*vphalf*Dtemp      & 
                          & /sqrt(vphalf**2 + 2*mugr(j)*bn(i))
                  end if
                  
                  jdum=j
                  kdum=k+1
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then                    
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.5*fac/dvp
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then        
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.5*fac/dvp 
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  if((k.eq.1).and.mass_conserve.and.lproc_vpar_lowerb)then
                     fac = 0.E0
                  else
                     call calfv(vmhalf,mugr(j)*bn(i),is,Dtemp) 
                     fac = 1.E0*vthrat(is)*vmhalf*Dtemp      & 
                          & /sqrt(vmhalf**2 + 2*mugr(j)*bn(i))
                  end if
                  
                  jdum=j
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then                    
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.5*fac/dvp
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k-1
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then        
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.5*fac/dvp 
                     call put_element(iih,jjh,mat_elem)
                  end if

                  !Second is the dmu term - Second order difference on a non-uniform grid
                  !requires all three terms due to lack of cancellation of f0 term.
             
                  if((j.eq.nmu).and.mass_conserve.and.lproc_mu_upperb)then
                     fac = 0.E0
                  else
                     call calfv(vpgr(i,j,k),muphalf*bn(i),is,Dtemp) 
                     fac = 2.E0*vthrat(is)*muphalf*Dtemp      & 
                          & /sqrt(vpgr(i,j,k)**2 + 2*muphalf*bn(i))
                  end if
                  
                  jdum=j+1
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then                    
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.5*fac/dmu
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then        
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = 0.5*fac/dmu 
                     call put_element(iih,jjh,mat_elem)
                  end if
                  
                  call calfv(vpgr(i,j,k),mumhalf*bn(i),is,Dtemp) 
                  fac = 2.E0*vthrat(is)*mumhalf*Dtemp      & 
                       & /sqrt(vpgr(i,j,k)**2 + 2*mumhalf*bn(i))
            
                  jdum=j
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then                    
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.5*fac/dmu
                     call put_element(iih,jjh,mat_elem) 
                  end if
                  
                  jdum=j-1
                  kdum=k
                  call frictionboundary(i,jdum,kdum,ingrida)
                  if(ingrida)then        
                     jjh = indx(imod,ix,i,jdum,kdum,is)
                     mat_elem = -0.5*fac/dmu 
                     call put_element(iih,jjh,mat_elem)
                  end if

               end do
            end do
         end do
         
      end do
   end do
end do

return 

end subroutine collision_friction_uniform

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine findvparpoint(vref,ii,jj,kk,duml,dumr,updo,ingridl,ingridr) 
!Function called by the collision operator when the grid is non-uniform
!it looks for the points to interpolate between
use velocitygrid, only : vpgr
use grid,         only : ns,nmu,nvpar,n_vpar_grid
use control,      only : vp_trap

!Reference values fed in from main program
integer ii,jj,kk
integer duml,dumr,updo,found
real    vref
logical :: ingridl,ingridr,ingrid_vpar

!updo is a parameter set to +1 looks in the row above, while -1 looks at
!the line below
if(vpgr(ii,jj,kk).eq.(0.E0))then
   duml = 0
   dumr = 0
   ingridl = .false.
   ingridr = .false.
   return
end if
if((jj.eq.nmu).and.(updo.eq.1))then
   duml = -1
   dumr = -1
   ingridl = .false.
   ingridr = .false.
   return
end if
if((jj.eq.1).and.(updo.eq.(-1)))then
   duml = 1
   dumr = 1
   ingridl = .false.
   ingridr = .false.
   return
end if
!Cant find points within the grid if looking above or below the domain and 
!points set to zero are ignored as they are not involved in the calculation
found=0
dumr=1
do while(found.eq.0)
   if((vpgr(ii,jj+updo,dumr).gt.vref).and.(vpgr(ii,jj+updo,dumr).ne.(0.E0)))then
      found=1
      ingridr = .true.
   else
      dumr=dumr+1
      if(dumr.gt.n_vpar_grid)then
         ingridr = .false.
         dumr = n_vpar_grid+1
         found = 1
      end if
   end if
end do
found=0
if(dumr.eq.(n_vpar_grid+1))then
   duml=n_vpar_grid
else 
   duml=dumr-1
end if
do while(found.eq.0)
   if((vpgr(ii,jj+updo,duml).lt.vref).and.(vpgr(ii,jj+updo,duml).ne.(0.E0)))then
      found=1
      ingridl = .true.
   else
      duml=duml-1
      if(duml.lt.1)then
         duml= 0
         ingridl = .false.
         return
      end if
   end if
end do
return

end subroutine findvparpoint

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine vgridboundary(ii,jj,kk,ingrid)

!APS: use velocitygrid, only : vpgr,mugr
use grid,      only : nvpar,nmu,n_mu_grid,n_vpar_grid

integer, intent(in) :: ii
integer, intent(inout) :: jj,kk
!The prefactors for the mu boundary conditions.
logical, intent(out) :: ingrid
real :: dmu

if (kk .gt. nvpar .and. lproc_vpar_upperb) then
   ingrid = .false.
   kk=nvpar
   if(jj.lt.1 .and. lproc_mu_lowerb)then
      ingrid=.false.
      return
   else if(jj.gt.nmu .and. lproc_mu_upperb)then
      ingrid=.false.
      return
   else
      return
   end if
else if (kk .lt. 1 .and. lproc_vpar_lowerb) then
   ingrid = .false.
   kk=1
   if(jj.lt.1 .and. lproc_mu_lowerb)then
      ingrid=.false.
      return
   else if(jj.gt.nmu .and. lproc_mu_upperb)then
      ingrid=.false.
      return
   else
      return
   end if
else if (jj .gt. nmu .and. lproc_mu_upperb) then
   ingrid = .false.
   jj=n_mu_grid
   if(kk.lt.1 .and. lproc_vpar_lowerb)then
      ingrid=.false.
      return
   else if(kk.gt.nvpar .and. lproc_vpar_upperb)then
      ingrid=.false.
      return
   else
      return
   end if
else if (jj .lt. 1 .and. lproc_mu_lowerb) then
   ingrid = .true.
   jj=1
   if(kk.lt.1 .and. lproc_vpar_lowerb)then
      ingrid=.false.
      return
   else if(kk.gt.nvpar .and. lproc_vpar_upperb)then
      ingrid=.false.
      return
   else
      return
   end if
else
   ingrid = .true.
   return
end if
call gkw_abort('Error in vgridboundary')

end subroutine vgridboundary

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine frictionboundary(ii,jj,kk,ingrid)

use grid,      only : nvpar,nmu

integer, intent(in) :: ii
integer, intent(inout) :: jj,kk
!The prefactors for the mu boundary conditions.
logical, intent(out) :: ingrid
real :: dmu

if (kk .gt. nvpar .and. lproc_vpar_upperb) then
   ingrid = .false.
   return
else if (kk .lt. 1 .and. lproc_vpar_lowerb) then
   ingrid = .false.
   return
else if (jj .gt. nmu .and. lproc_mu_upperb) then
   ingrid = .false.
   return
else if (jj .lt. 1 .and. lproc_mu_lowerb) then
   ingrid = .true.
   jj=1
   return
else
   ingrid = .true.
   return
end if
call gkw_abort('Error in frictionboundary')

end subroutine frictionboundary

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine momentumboundary(jj,kk,ingrid)

use grid,      only : nvpar,nmu

integer, intent(inout) :: jj,kk
!The prefactors for the mu boundary conditions.
logical, intent(out) :: ingrid
real :: dmu

if (kk .gt. nvpar .and. lproc_vpar_upperb) then
   ingrid = .false.
   kk = nvpar
   return
else if (kk .lt. 1 .and. lproc_vpar_lowerb) then
   ingrid = .false.
   kk = 1
   return
else if (jj .gt. nmu .and. lproc_mu_upperb) then
   ingrid = .false.
   jj = nmu
   return
else if (jj .lt. 1 .and. lproc_mu_lowerb) then
   ingrid = .true.
   jj=1
   return
else
   ingrid = .true.
   return
end if
call gkw_abort('Error in frictionboundary')

end subroutine momentumboundary

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine secondordernonuniform(vref,ii,jj,kk,updo,inta,intb,intc,a,b,c,ingrida,ingridb,ingridc)
!----------------------------------------------------------------
!a,b,c are the coefficients containing the parallel velocity element
!information while ingrida,b,c tells us whether the point in question
!is within the confines of the grid.
!Uses parts of findvparpoint to find points to difference between
!and interpolate.
!----------------------------------------------------------------

use grid,     only : nmod,nx,ns,nmu,nvpar,nsp
use control,  only : vp_trap
use velocitygrid, only : mugr,vpgr,dvp

implicit none

logical ingrida,ingridb,ingridc
real a,b,c,vref
integer updo,inta,intb,intc,ii,jj,kk
real dv1,dv2,dv3

!Finds the points to difference between
!The integer updo looks in the line above when +1 and the line
!below when -1, if 0 looks at the line (same mu) of vref
call findvparpoint(vref,0,jj,kk,inta,intb,updo,ingrida,ingridb)
dv1 = vref - vpgr(ii,jj,inta)
dv2 = vref - vpgr(ii,jj,intb)
if(dv1.gt.dv2)then
   intc = intb+1
else if(dv1.lt.dv2)then
   intc = inta-1
else
   stop 'Some error secondordernonuniform'
end if
dv1 = vref - vpgr(ii,jj,inta)
dv2 = vpgr(ii,jj,intb)-vref
!Need to decide here if middle point is to the right or left of vref
dv3 = 0

c = dv1*dv1/((dv2*dv2*dv1) + (dv1*dv1*dv2))
b = dv3/((dv1+dv3)*((dv2*dv2*dv1) + (dv1*dv1*dv2)))
a = -((dv2*dv2) - (dv1*(dv2*dv2-dv1*dv1)/(dv1+dv3)))/(dv2*dv2*dv1+dv1*dv1*dv2)

end subroutine secondordernonuniform

!----------------------------------------------------------------
!----------------------------------------------------------------

subroutine twopointinterp(vref,ii,jj,kk,updo,inta,intb,a,b,ingrida,ingridb)

use grid,     only : nmod,nx,ns,nmu,nvpar,nsp
use control,  only : vp_trap
use velocitygrid, only : mugr,vpgr,dvp

implicit none

logical ingrida,ingridb
real a,b,c,vref
integer updo,inta,intb,ii,jj,kk
real dv1,dv2

call findvparpoint(vref,0,jj,kk,inta,intb,updo,ingrida,ingridb)

   dv1 = vref-vpgr(ii,jj,inta)
   dv2 = vpgr(ii,jj,intb)-vref

a = dv2/(dv1+dv2)
b = dv1/(dv1+dv2)

return

end subroutine twopointinterp

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine cons_momentum
!--------------------------------------------------------------------
! implements the simple model for momentum conservation
! The integrals are performed in exp_integration to save time
!
!-------------------------------------------------------------------

use grid,   only : nmod,nx,ns,nmu,nvpar,nsp
use velocitygrid,   only : mugr,vpgr, intvp, intmu
use dist,       only : indx,fmaxwl,i_mom,fdisi
use geom,       only : bn 
use components, only : vthrat
use matdat,     only : put_element 

implicit none

integer i,j,k,ix,imod,is
complex dum, mat_elem
integer iih,jjh

do is=1,nsp;do ix=1,nx;do imod=1,nmod
   
   do i=1,ns
      do j=1,nmu
         do k = 1,nvpar
            iih = indx(imod,ix,i,j,k,is)
            jjh = indx(imod,ix,i,i_mom,is)
            mat_elem = -fmaxwl(i,j,k)*vpgr(i,j,k)/sqrt(vpgr(i,j,k)**2 + 2.E0*mugr(j)*bn(i))
           call put_element(iih,jjh,mat_elem) 
         end do
      end do
   end do
      
end do;end do;end do

return
end subroutine cons_momentum 

!Calculatio of the prefactors of the integrals in the 
!momentum conserving term
subroutine coll_mom_change_int

use mpiinterface
use mpicomms
use grid,   only : nmod,nx,ns,nmu,nvpar,nsp
use velocitygrid,   only : mugr,vpgr, intvp, intmu, dvp
use dist,       only : indx,fmaxwl,fdisi,i_mom
use geom,       only : bn 
use components, only : vthrat
use constants,  only : pi
use matdat,  only : put_element

implicit none

integer i,j,k,ix,imod,is,ierr
integer kdum,jdum
complex dumtheta,dumnu,dumfv,dum2,dum
integer pp
real Dtemp, dmup
real Dtt, Dnu, Dfu
real mumhalf, muphalf, vphalf, vmhalf
real :: dmud
logical ingrid
complex :: mat_elem
integer :: iih, jjh

  do i=1,ns
  
    ! For each i, calculate the integral over the maxwellian first.
    dum  = 0.
    do j=1,nmu
      do k=1,nvpar
        dum = dum + intvp(i,j,k)*intmu(j)*fmaxwl(i,j,k)*vpgr(i,j,k)**2/    &
             & sqrt(vpgr(i,j,k)**2 + 2.E0*mugr(j)*bn(i))
      end do
    end do
    ! reduce over points with the same every apart from vpar and mu
    call mpiallreduce_sum(dum,dum2,1,COMM_VPAR_NE_MU_NE)
    
    do is=1,nsp ; do ix=1,nx ; do imod = 1,nmod
    
      iih = indx(imod,ix,i,i_mom,is)
      
      do j=1,nmu ; do k=1,nvpar
                  
        !Precalculate the diffusion coefficients for each of the 
        !three collision operator terms
        if (pitch_angle) then
          call selfcaldthth(vpgr(i,j,k),mugr(j)*bn(i),is,Dtemp)
          Dtt = 2.*mugr(j)*Dtemp /(vpgr(i,j,k)**2 + 2.*mugr(j)*bn(i))                
        else
          Dtt = 0.
        end if
        
        if (en_scatter) then
          call selfcaldvv(vpgr(i,j,k),mugr(j)*bn(i),is,Dtemp)
          Dnu = 1.*vpgr(i,j,k)*Dtemp /(vpgr(i,j,k)**2 + 2.*mugr(j)*bn(i))
        else
          Dnu = 0.
        end if
        
        if (friction_coll) then
           call selfcalfv(vpgr(i,j,k),mugr(j)*bn(i),is,Dtemp)
           Dfu = 1.*vpgr(i,j,k)*Dtemp/sqrt(vpgr(i,j,k)**2 + 2.*mugr(j)*bn(i))  

           !The term for the friction conservation
           jjh = indx(imod,ix,i,j,k,is)
           
           mat_elem = Dfu*intvp(i,j,k)*intmu(j)/dum2
           call put_element(iih,jjh,mat_elem)
        end if
           
        !The mu intervals above and below the point of
        !interest
        
        vphalf = vpgr(i,j,k) + 0.5*dvp 
        vmhalf = vpgr(i,j,k) - 0.5*dvp
        
        !The mu intervals above and below the point of
        !interest
        
        if(j.eq.1 .and. lproc_mu_lowerb)then
           dmup = mugr(2)-mugr(1)
           dmud = 2.E0*mugr(1)
        else
           dmup = mugr(j+1)-mugr(j)
           dmud = mugr(j)-mugr(j-1)
        end if
        
        !The half grid mu values
        muphalf = mugr(j)+0.5*dmup
        mumhalf = mugr(j)-0.5*dmud
        
        jdum = j
        kdum = k+1
        call momentumboundary(jdum,kdum,ingrid)
        if(ingrid)then
           jjh = indx(imod,ix,i,j,kdum,is)
           mat_elem = 0.5*(Dtt*bn(i)+Dnu*vpgr(i,j,k))/(dum2*dvp)                   
        else
           jjh = indx(imod,ix,i,j,k,is)
           mat_elem = -0.5*(Dtt*bn(i)+Dnu*vpgr(i,j,k))/(dum2*dvp)
        end if
                           
        mat_elem = mat_elem*intvp(i,j,k)*intmu(j)
        !write(*,*)iih,jjh,mat_elem
        call put_element(iih,jjh,mat_elem)

        jdum = j
        kdum = k-1
        call  momentumboundary(jdum,kdum,ingrid)
        if(ingrid)then
           jjh = indx(imod,ix,i,j,kdum,is)
           mat_elem = -0.5*(Dtt*bn(i)+Dnu*vpgr(i,j,k))/(dum2*dvp)  
        else
           jjh = indx(imod,ix,i,j,k,is)
           mat_elem = 0.5*(Dtt*bn(i)+Dnu*vpgr(i,j,k))/(dum2*dvp)  
        end if
          
        mat_elem = mat_elem*intvp(i,j,k)*intmu(j)
        call put_element(iih,jjh,mat_elem)

        jdum = j+1
        kdum = k
        call  momentumboundary(jdum,kdum,ingrid)
        if(ingrid)then
           jjh = indx(imod,ix,i,jdum,k,is)
           mat_elem = 0.5*(-Dtt*vpgr(i,j,k)+2.E0*mugr(j)*Dnu)/               &
                    &     (dum2*(muphalf-mumhalf))
        else
           jjh = indx(imod,ix,i,j,k,is)
           mat_elem = -0.5*(-Dtt*vpgr(i,j,k)+2.E0*mugr(j)*Dnu)/              &
                    &      (dum2*(muphalf-mumhalf))
        end if                 
         
        mat_elem = mat_elem*intvp(i,j,k)*intmu(j)
        call put_element(iih,jjh,mat_elem) 
        
        kdum = k
        jdum = j-1
        call  momentumboundary(jdum,kdum,ingrid)
        if(ingrid)then
           jjh = indx(imod,ix,i,jdum,k,is)
           mat_elem = -0.5*(-Dtt*vpgr(i,j,k)+2.E0*mugr(j)*Dnu)/              &
                    &      (dum2*(muphalf-mumhalf))
        else
           jjh = indx(imod,ix,i,j,k,is)
           mat_elem = 0.5*(-Dtt*vpgr(i,j,k)+2.E0*mugr(j)*Dnu)/               &
                    &     (dum2*(muphalf-mumhalf))
        end if

        mat_elem = mat_elem*intvp(i,j,k)*intmu(j)
        call put_element(iih,jjh,mat_elem)

      end do ; end do
      
    end do ; end do ; end do

  end do

end subroutine coll_mom_change_int


subroutine coll_mom_change_diag

use matdat,  only : put_element
use dist,    only : i_mom, indx
use grid,   only : nmod,nx,ns,nsp

implicit none

integer :: ix, i, is, imod
complex :: mat_elem
integer :: iih, jjh

do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns; do is = 1, nsp

    ! reference the element of the potential
    iih = indx(imod,ix,i,i_mom,is)
    jjh = iih    
    ! initialize the mat_element
    mat_elem = (1.,0.)
    call put_element(iih,jjh,mat_elem)

end do; end do; end do; end do

end subroutine coll_mom_change_diag

!****************************************************************************
!> Derivative of erf  2*exp(-x^2)/ sqrt(\pi)
!----------------------------------------------------------------------------
function erfp(x)

  real, intent(in) :: x
  real :: erfp
  real, parameter :: two_over_root_pi = 1.1283791670955125738961589E0
     
  erfp = two_over_root_pi*exp(-x**2)
  return
  
end function erfp

!****************************************************************************
!> get the global species number
!----------------------------------------------------------------------------

function gsn(local_species_number)

  use grid, only : isppb
  
  integer, intent(in) :: local_species_number
  integer :: gsn
  
  gsn = isppb + local_species_number - 1

end function gsn

!****************************************************************************
  
end module collisionop
