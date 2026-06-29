!----------------------------------------------------------------------------
! SVN: $Id: constants.f90 1005 2009-07-02 16:12:03Z  $
!> Contains some physical and mathematical constants
!----------------------------------------------------------------------------
module constants 

  implicit none

!
! ========== mathematical constants ==========
!

  !>\pi
  real, parameter :: pi =  3.1415926535897932384626433832795028841971693993751

  !> The imaginary number i=sqrt(-1) 
  complex, parameter :: ci1 = (0.,1.)
  
!
! =========== physical constants ==========
!

  ! The mass of a proton 
  real, parameter :: proton_mass = 1.67262158E-27

  ! epsilon_0
  real, parameter :: epsilon_0 = 8.85418782E-12

  ! The unit of charge 
  real, parameter :: unit_charge_si = 1.602176487E-19

end module constants 
