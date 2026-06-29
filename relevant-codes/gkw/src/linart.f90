! SVN:$Id: linart.f90 1009 2009-07-02 17:08:44Z  $
!> main program
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
program linart

!----------------------------------------------------------------------------
! Normalization: all length scales are normalized
! to the major radius. All velocities are normalized
! the the thermal velocity of the ions. 
! NOTE vth = sqrt(2 T / m )
!
! CALLS
! mpiinit           : module mpiinterface : file mpiinterface.F90
! initialize        : module init         : file init.f90
! write_output      : module diagnostic   : file diagnostic.F90 
! calc_linear_terms : module linear_terms : file linear_terms.F90
! compress_matrix   : module matdat       : file matdat.F90
! final_output      : module diagnostic   : file diagnostic.F90
! cpu_time          : intrinsic function 
! 
! GLOBAL variables
! testing           : module control      : file control.f90
!
! LOCAL variables 
! t_begin           : time at begin of the run 
! t_end             : time at the end of the run 
!--------------------------------------------------------------------

  use mpiinterface
  use control,         only : ntime, naverage, max_seconds, lfinal_output,   &
                            & lwrite_output1, lcalc_fluxes, method, max_sec, &
                            & stop_me 
  use linear_terms,    only : calc_linear_terms
  use diagnostic,      only : output_init, output_finalize,write_output,     &
                            & final_output, fluxes 
  use matdat,          only : compress_matrix
  use exp_integration, only : explicit_integration
  use init,            only : initialize, deallocate_runtime_arrays
  use general,         only : gkw_abort

  implicit none

  integer :: i
  double precision :: t_begin, t_end, t_tus, t_1, t_predict
  double precision :: t_begin_main, t_end_main
  logical :: exstop

  ! initialize mpi
  call mpiinit()

  ! get the start time of the run 
  t_begin = MPI_WTIME()

  ! call the initialization
  call initialize
   
  ! set up the files for output
  call output_init

  ! calculate the linear terms 
  call calc_linear_terms
  
  ! call the matrix orderer
  call compress_matrix
  
  ! get the time at the start of the main loops
  t_begin_main = MPI_WTIME()
  
  ! loop over large time steps
  large_time_steps : do i = 1, ntime

    select case(method) 
    case('EXP')
      ! do the time integration naverage times 
      call explicit_integration(i)
    case default 
      call gkw_abort('Unknown method of integration')
    end select
 
    ! calculate the fluxes 
    if (lcalc_fluxes) call fluxes

    ! write the small output  
    if (lwrite_output1) call write_output

    ! check for external stop criterium
    inquire(file='gkw.stop',exist = exstop) 
    if (exstop) then
      call mpibarrier()
      if (root_processor) then
        write(*,*)'External stop '
        open (9, FILE = 'gkw.stop')
        close (9, STATUS='delete')              
      end if
      exit large_time_steps
    end if 
    
    !Predict total runtime after first iteration
    if (i == 1 .and. root_processor) then
      
      !Time for one iteration in seconds
      t_1 = MPI_WTIME()-t_begin_main
      !Total predicted runtime in seconds
      t_predict=t_1*ntime

      if (t_predict > 60.) then
        write(*,*)
        write(*,*) 'Iteration 1 completed sucessfully.'
        write(*,*) 'Predicted runtime: ', nint(t_predict/60.), ' minutes'
        write(*,*)
      end if
      
      if (t_predict > max_seconds .and. max_seconds > 0.) then
        write(*,*) 'WARNING: Run likely to terminate from max_sec input'
        write(*,*) 'Stop will occur after ', nint((max_sec-300)/60.), 'minutes' 
        write(*,*) 'Iterations expected: ', nint((max_sec-300)/t_1), 'of ',  &
            &       ntime,' requested'
        write(*,*)
      end if
    end if

    ! check if maximum time has been exceeded 
    if (max_seconds > 0.) then 
      t_tus = MPI_WTIME()
      ! the  standard is to stop 5 minutes before 
      if (t_tus-t_begin > max_seconds - 300.) then
        if (root_processor) then 
          write(*,*) 'max_sec: stop' 
        end if
        exit large_time_steps
      end if 
    end if 

    ! check if the code has reached convergence and wants to stop 
    if (stop_me) then 
      if (root_processor) then 
        write(*,*)'Convergence reached : stop' 
      endif 
      exit large_time_steps 
    endif 

  end do large_time_steps

  ! get the main loop end time
  t_end_main = MPI_WTIME()

  ! deallocate arrays used only in the main time loop
  call deallocate_runtime_arrays
  
  if (lfinal_output) call final_output

  ! clean up any output and close files
  call output_finalize
  
  ! determine the CPU time used 
  t_end = MPI_WTIME()

  if (root_processor) then
    ! Please do not remove the success message from the following line
    ! unless the test script is modified accordingly.
    write(*,*)'Run succesfully completed, Run time :',t_end-t_begin 
    write(*,*)'                    (main loop time):',t_end_main-t_begin_main  
  end if   

  ! finilize mpi 
  call mpifinalize()

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end program linart
