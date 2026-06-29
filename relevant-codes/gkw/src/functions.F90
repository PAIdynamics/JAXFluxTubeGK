! $Id: functions.F90 1005 2009-07-02 16:12:03Z  $
!> i don't have a specific purpose right now
!----------------------------------------------------------------------------

module functions

  use mpiinterface
  use mpidatatypes

  private

  public :: gamma_gkw, besselj0_gkw, norm

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This function calculates the Gamma function needed in Poissons and
!> Ampere's equation
!----------------------------------------------------------------------------

function gamma_gkw(imod,ix,i,is)

  use specfun,    only : expbessi0
  use components, only : rhorat
  use geom,       only : bn
  use mode,       only : krloc

  integer, intent(in) :: imod, ix, i, is

  integer :: j, k
  real :: gamma_gkw, dum1, gamma_num, b

  dum1 = 0.5*(rhorat(is)*krloc(imod,ix,i)/bn(i))**2
  gamma_gkw = expbessi0(dum1)

end function gamma_gkw

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This function calculates the bessel function J0 used in the Gyro average
!----------------------------------------------------------------------------

function besselj0_gkw(imod,ix,i,j,is)

  use specfun,      only : bessj0
  use components,   only : rhorat 
  use velocitygrid, only : mugr
  use geom,         only : bn
  use mode,         only : krloc

  integer, intent(in) :: imod, ix, i, j, is 

  real :: besselj0_gkw

  besselj0_gkw = bessj0(rhorat(is)*krloc(imod,ix,i)*                         & 
               & sqrt(2.E0*mugr(j)/bn(i)))

end function besselj0_gkw

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module functions
