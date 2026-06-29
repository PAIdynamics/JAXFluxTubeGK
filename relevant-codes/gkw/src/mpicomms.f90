!> somewhere to keep the additional MPI communicators
module mpicomms

  implicit none
  
  !> a communicator incorporating all processors
  integer :: COMM_CART  
  
  !> procs for the same points in all directions other than s
  integer :: COMM_S_NE
  !< all procs responsible for the same points in s
  integer :: COMM_S_EQ
  
  !> procs for the same points in all directions other than vpar
  integer :: COMM_VPAR_NE
  !> procs for the same points in all directions other than vpar and mu
  integer :: COMM_VPAR_NE_MU_NE
  !> all procs responsible for the same points in vpar
  integer :: COMM_VPAR_EQ
  
  !> procs for the same points in all directions other than mu
  integer :: COMM_MU_NE
  !> all procs responsible for the same points in mu
  integer :: COMM_MU_EQ
  
  !> procs for the same points in all directions, but different species
  integer :: COMM_SP_NE
  !> all procs responsible for the same species
  integer :: COMM_SP_EQ

  !> intersection of COMM_SP_EQ and COMM_S_EQ
  integer :: COMM_SP_EQ_S_EQ
  
  !> a communicator for self
  integer :: COMM_ALL_EQ

end module mpicomms
