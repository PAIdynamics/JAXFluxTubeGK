!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!  $Id: global.f90 1023 2009-07-02 20:27:02Z phsgbq $
!> Various things needed in most parts of the code
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

module global

  implicit none

  private
 
  !
  ! Parammeters
  !
  
  !> tag
  integer, parameter, public :: I_RUN_WITHOUT_MPI = -64523

  !> id for vpar
  integer, parameter, public :: id_vpar  = 1
  !> id for mu
  integer, parameter, public :: id_mu    = 2
  !> id for s
  integer, parameter, public :: id_s     = 3
  !> id for mod
  integer, parameter, public :: id_mod   = 4
  !> id for x
  integer, parameter, public :: id_x     = 5
  !> id for species
  integer, parameter, public :: id_sp    = 6
  !> dummy id
  integer, parameter, public :: id_dummy = -7
  
  !> length for characters
  integer, parameter, public :: lenswitch = 32
  !> large integer
  integer, parameter, public :: i_huge = huge(1)/5
  !> some large integer
  integer, parameter, public :: i_huge_tag = 62768-huge(1)/5
  !> large real
  real, parameter,    public :: r_huge = 0.2*huge(1.)
  !> small real
  real, parameter,    public :: r_tiny = 5.*tiny(1.)
 
  character (len=32),  parameter, public :: GKW_REV = 'CPC_release'
  character (len=32),  parameter, public :: GKW_EXE = 'gkw.x'
  character (len=128), parameter, public :: GKW_FC =  'fortran_compiler'

  !
  ! Global variables
  !

  logical, public :: lverbose = .false.
  
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module global
