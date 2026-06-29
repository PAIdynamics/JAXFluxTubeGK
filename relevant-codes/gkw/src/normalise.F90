! $Id: normalise.F90 1008 2009-07-02 16:44:23Z  $
!----------------------------------------------------------------------------
!> Routines related to the the normalisation of fdis.
!> The different switches for the normalize routine are:
!>   isw = -1 ; calculate the change in phase (for the real_frequency)
!>   isw =  1 ; only calculate the normalisation factor
!>   isw =  2 ; calculate the normalisation factor and perform the
!>              normalisation of fdis (also calculate the growth_rate and
!>              real_frequency)
!>   isw =  3 ; only normalise fdis with the normalisation factor 
!----------------------------------------------------------------------------

module normalise

  implicit none
  private

  public :: normalize
 
  !> The growth rate of the mode.
  real, public :: growth_rate
  !> The real frequncy of the mode.
  real, public :: real_frequency

  ! retained variables for the calculation of the real_frequency
  real,save :: last_phase_time = 0.
  real,save :: delta_time = 1.
  ! the normalisation factor
  real :: factor
  ! The arrays to keep track of the phase for the mode frequency calculation
  real :: phase, last_phase, last_growth 
  
contains

!****************************************************************************
!****************************************************************************

subroutine normalize(isw,fdis,nsolc)  ! Optimized routine 

  use control, only : non_linear, normalized
  
  integer, intent(in) :: isw, nsolc
  complex, intent(inout) :: fdis(nsolc)

  ! Never normalize a nonlinear run 
  if ((non_linear).or.(.not.normalized)) return 

  ! The normalization factor is not calculated for 
  ! the case isw = 3 (only normalization)
  if (isw /= 3) then 
    if (isw == -1) then
      ! calculates the phase
      call calc_phase(fdis,nsolc)
    else
      call calc_factor(fdis,nsolc)
    end if
  end if

 
  if (isw == 1 .or. isw == -1) return

  ! re-scale the distribution function and the potential with the factor
  fdis = fdis / factor

  ! calculate the growth rate and frequency
  if (isw == 2) call growth_rate_and_freq

end subroutine normalize 

!****************************************************************************

subroutine calc_phase(fdis,nsolc)

  use mpiinterface
  use mpicomms
  use global,    only : r_tiny
  use constants, only : pi
  use control,   only : time
  use grid,      only : parallel_s
  use dist,      only : nphi

  integer, intent(in) :: nsolc
  complex, dimension(nsolc), intent(in) :: fdis
!  complex :: tmp
  
  real :: tmp1, tmp2, reduced
  integer :: ierr
  
  ! save time interval and update last_phase_time
  delta_time = time - last_phase_time
  last_phase_time = time

  ! before calculating the new phase, store the old one
  last_phase = phase

  ! All the fields are used, calculation works also when the code is run 
  ! on more than one processor

  tmp1 = sum(real(fdis(nphi:nsolc)))
  tmp2 = sum(aimag(fdis(nphi:nsolc)))

  ! sum-reduce the over the s-direction
#ifdef mpi
    if (parallel_s) then
      ierr = 0
      call MPI_ALLREDUCE(tmp1,reduced,1,MPIREAL_X,MPI_SUM,COMM_S_NE,ierr)
      tmp1 = reduced
      call MPI_ALLREDUCE(tmp2,reduced,1,MPIREAL_X,MPI_SUM,COMM_S_NE,ierr)
      tmp2 = reduced
    end if
#endif

   phase = atan2(tmp2,tmp1)

  ! Treat the cases of 2\pi jumps. 
  if (abs(phase-last_phase) > max(pi/4.,abs(3.*real_frequency*delta_time))) then
      last_phase = last_phase - sign(2*pi,last_phase)
  end if

end subroutine calc_phase

!****************************************************************************
!> Calculate a normalisation factor for fdisi based on the fields; the sqrt
!> of the sum over the fields and the s-direction of the abs^2 field values.
!----------------------------------------------------------------------------

subroutine calc_factor(fdis,nsolc)

  use mpiinterface
  use mpicomms
  use global,  only : r_tiny
  use control, only : nlapar
  use grid,    only : parallel_s
  use dist,    only : nphi
  
  integer, intent(in) :: nsolc
  complex, dimension(nsolc), intent(in) :: fdis

  real :: reduced
  integer :: ierr
   
  ! local sum over the fields nphi:nsolc
  factor = sum(abs(fdis(nphi:nsolc))**2)
  
  ! At first time step the potential is not necessarily initialized 
  if (factor < r_tiny) then
    factor = 1.
  else
    ! for any spatial parallelization, reduce-sum the factor over space
#ifdef mpi
    if (parallel_s) then
      ierr = 0
      call MPI_ALLREDUCE(factor,reduced,1,MPIREAL_X,MPI_SUM,COMM_S_NE,ierr)
      factor = reduced
    end if
#endif
  end if

  factor = sqrt(factor)

end subroutine calc_factor

!****************************************************************************
!> calculate the growth rate and real frequency
!----------------------------------------------------------------------------

subroutine growth_rate_and_freq

  use global,  only : r_tiny
  use control, only : time, lcalc_freq, gamatol, stop_me 
  use mode,    only : mode_box
  
  ! last_time is retained so that the growth rate can be calculated.
  real, save :: last_time = 0.
  
  if (abs(time-last_time) < r_tiny) then
    last_growth = 0. 
    growth_rate = 0.
    if (lcalc_freq) real_frequency = 0.
  else
    ! calculate the growth rate 
    last_growth = growth_rate 
    growth_rate = log(factor) / (time - last_time)
    

    ! do the test on the growth rate 
    if (abs((last_growth - growth_rate)/(time-last_time)).lt.gamatol) &
       & stop_me = .true.

    if (lcalc_freq) then
      ! calculate the frequency
      real_frequency = (phase - last_phase) / delta_time
    else
      ! not calculated at the moment
      real_frequency = 0.  
    end if
  end if
  
  ! reset the time
  last_time = time 

end subroutine growth_rate_and_freq

!****************************************************************************
!****************************************************************************

end module normalise
