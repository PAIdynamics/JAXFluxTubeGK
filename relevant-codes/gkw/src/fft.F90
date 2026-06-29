! SVN:$Id: fft.F90 1023 2009-07-02 20:27:02Z phsgbq $
#if defined(fft_w2)
MODULE fft
!
  IMPLICIT NONE
!
  PRIVATE
!
  PUBLIC four1, fourcol, fourrow, four1_real
!
! Global parameters
!
  INTEGER, PARAMETER :: MXPLAN=8
!
! Global variables
!
  INTEGER ::n1d_saved=0
  INTEGER*8, DIMENSION(MXPLAN)             :: plan1d
  INTEGER,   DIMENSION(MXPLAN)             :: n1d
  REAL,      DIMENSION(:),     ALLOCATABLE :: scr1_real
  COMPLEX,   DIMENSION(:),     ALLOCATABLE :: scr1

  INTERFACE fourcol
     MODULE PROCEDURE fourcol_ra, fourcol_raa
  END INTERFACE

CONTAINS
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four1(arr, isign)
!
!  A single 1D complex FFT
!
    INCLUDE 'fftw_f77.h'
!
! Dummy arguments
!
!!$    DOUBLE COMPLEX, DIMENSION(:), INTENT(INOUT) :: arr
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: arr
    INTEGER,               INTENT(IN)    :: isign
!
! Local variables
!
    INTEGER :: n, id
!
    n = SIZE(arr)
    IF( .NOT. ALLOCATED(scr1) ) THEN
       ALLOCATE(scr1(n))
    ELSE
       IF ( SIZE(scr1) < n ) THEN
          DEALLOCATE(scr1)
          ALLOCATE(scr1(n))
       END IF
    END IF
!
    CALL getplan(n, isign, id, 1)
    CALL fftw_f77_one(plan1d(id), arr, scr1)
  END SUBROUTINE four1
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four1_real(arr, isign)
!
!  A single 1D real FFT
!  Elements are rearranged to pack modes (m,-m) together. 
!
    IMPLICIT NONE
    INCLUDE 'fftw_f77.h'
    REAL, DIMENSION(:), INTENT(INOUT) :: arr
    INTEGER, INTENT(IN) :: isign
    INTEGER :: n, id,i
!
    n = SIZE(arr)
    IF( .NOT. ALLOCATED(scr1_real) ) THEN
       ALLOCATE(scr1_real(n))
    ELSE
       IF ( SIZE(scr1_real) < n ) THEN
          DEALLOCATE(scr1_real)
          ALLOCATE(scr1_real(n))
       END IF
    END IF
!
    CALL getplan(n, isign, id,0)
    IF ((.TRUE.).AND.(isign == 1)) THEN
       scr1_real = arr
       DO i=1,n/2-1
          arr(i+1) = scr1_real(i*2)
          arr(n+1-i) = scr1_real(i*2+1)
       END DO
       arr(1)   = scr1_real(1)
       arr(n/2+1) = scr1_real(n)
    END IF
    CALL rfftw_f77_one(plan1d(id), arr, scr1_real)
    IF ((.TRUE.).AND.(isign == -1)) THEN
       scr1_real = arr
       DO i=1,n/2-1
          arr(i*2)   = scr1_real(i+1)
          arr(i*2+1) = scr1_real(n+1-i)
       END DO
       arr(1) = scr1_real(1)
       arr(n) = scr1_real(n/2+1)
    END IF
  END SUBROUTINE four1_real
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_ra(arr, isign)
!
!  1D complex FFT of columns of arr(1:N,1:howmany)
!
     INCLUDE 'fftw_f77.h'
!
! Dummy arguments
!
!!$    DOUBLE COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr
    INTEGER,                 INTENT(IN)    :: isign
!
! Local variables
!
    INTEGER :: n, howmany, id
!
    n = SIZE(arr,1)
    howmany = SIZE(arr,2)
!
    IF( .NOT. ALLOCATED(scr1) ) THEN
       ALLOCATE(scr1(n))
    ELSE
       IF ( SIZE(scr1) < n ) THEN
          DEALLOCATE(scr1)
          ALLOCATE(scr1(n))
       END IF
    END IF
!
    CALL getplan(n, isign, id,1)
    CALL fftw_f77(plan1d(id), howmany, arr, 1, n, scr1, 1, n)
  END SUBROUTINE fourcol_ra
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_raa(arr, isign)
!
!  1D complex FFT of columns of arr(1:N,1:howmany)
!
     INCLUDE 'fftw_f77.h'
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:,:), INTENT(INOUT) :: arr
    INTEGER,                   INTENT(IN)    :: isign
!
! Local variables
!
    INTEGER :: n, howmany, id
!
    n = SIZE(arr,1)
    howmany = SIZE(arr,2)*SIZE(arr,3)
!
    IF( .NOT. ALLOCATED(scr1) ) THEN
       ALLOCATE(scr1(n))
    ELSE
       IF ( SIZE(scr1) < n ) THEN
          DEALLOCATE(scr1)
          ALLOCATE(scr1(n))
       END IF
    END IF
!
    CALL getplan(n, isign, id, 1)
    CALL fftw_f77(plan1d(id), howmany, arr, 1, n, scr1, 1, n)
  END SUBROUTINE fourcol_raa
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourrow(arr, isign)
!
!  1D complex FFT of rows of arr(1:howmany,1:N)
!
    INCLUDE 'fftw_f77.h'
!
! Dummy arguments
!
!!$    DOUBLE COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr
    INTEGER,                 INTENT(IN)    :: isign
!
! Local variables
!
    INTEGER :: n, howmany, id
!
    n = SIZE(arr,2)
    howmany = SIZE(arr,1)
!
    IF( .NOT. ALLOCATED(scr1) ) THEN
       ALLOCATE(scr1(n))
    ELSE
       IF ( SIZE(scr1) < n ) THEN
          DEALLOCATE(scr1)
          ALLOCATE(scr1(n))
       END IF
    END IF
!
    CALL getplan(n, isign, id, 1)
    CALL fftw_f77(plan1d(id), howmany, arr, howmany, 1, &
         & scr1, howmany, 1)
  END SUBROUTINE fourrow
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE getplan(n, sign, id, complex_fftw)
!
!  Create or get an already created FFT plan (depends only on N.)
!
    INCLUDE 'fftw_f77.h'
!
! Dummy arguments
!
    INTEGER, INTENT(IN)  :: n            ! size of transform
    INTEGER, INTENT(IN)  :: sign         ! dir. of transform -1=>FORWARD, +1=>BACKWARD
    INTEGER, INTENT(OUT) :: id           ! id of FFT plan
    INTEGER, INTENT(IN)  :: complex_fftw ! Create complex<->complex transform if =1
!
! Local variables
!
    INTEGER :: k, i, dir
!
    k = sign*(2*n+complex_fftw)
    DO i = 1,n1d_saved
       IF( k == n1d(i)) THEN
          id = i
          RETURN
       END IF
    END DO
    IF( n1d_saved == MXPLAN) THEN
       PRINT*, 'Module fft: MXPLAN too small! Increase it and recompile'
       STOP
    END IF
    n1d_saved = n1d_saved+1
    n1d(n1d_saved) = k
    id = n1d_saved
    dir = FFTW_FORWARD
    IF( sign == +1 ) dir = FFTW_BACKWARD
    IF (complex_fftw == 1) THEN
       CALL fftw_f77_create_plan(plan1d(id), n, dir, FFTW_MEASURE + FFTW_IN_PLACE)
    ELSE  
       CALL rfftw_f77_create_plan(plan1d(id), n, dir, FFTW_MEASURE + FFTW_IN_PLACE)
    END IF
  END SUBROUTINE getplan
END MODULE fft
#endif
!
!
!
#if defined(fft_w3)
!
! fftw_measure replaced by fftw_estimate 
! Fri Mar  9 13:43:53 CET 2007
!
MODULE fft
!
  IMPLICIT NONE
!
  PRIVATE

  !> if this interface compiles, FFTW should work
  LOGICAL, PUBLIC :: WORKING_FFT_LIBRARY=.true.
  !PUBLIC :: four1D_real, fourcol_real, fourrow_real
  PUBLIC :: four2D_real, fourcol
  !PUBLIC :: four1D, fourrow
!
  INCLUDE 'fftw3.f'
!
  TYPE int_para
     INTEGER, DIMENSION(2) :: par ! size of transform
  END TYPE int_para
!
! Global parameters
!
  INTEGER, PARAMETER :: MXPLAN=16 ! define the maximum number of plans.
!
! Global variables
!
  INTEGER*8,      DIMENSION(MXPLAN,8), SAVE :: plan1d      ! plans for 1-dim FFT
  TYPE(int_para), DIMENSION(MXPLAN,8), SAVE :: n1d_par
  INTEGER,        DIMENSION(8),        SAVE :: n1d_saved=0 ! number of plans saved
!
  INTEGER*8,      DIMENSION(MXPLAN,1), SAVE :: plan2d      ! plans for 2-dim FFT
  TYPE(int_para), DIMENSION(MXPLAN,1), SAVE :: n2d_par
  INTEGER,        DIMENSION(1),        SAVE :: n2d_saved=0 ! number of plans saved
!
!!!  INTERFACE four1D_real
!!!     MODULE PROCEDURE four1D_ra_ca
!!!  END INTERFACE
!!!!
!!!  INTERFACE fourcol_real
!!!     MODULE PROCEDURE fourcol_ra_ca, fourcol_raa_caa
!!!  END INTERFACE
!!!!
!!!  INTERFACE fourrow_real
!!!     MODULE PROCEDURE fourrow_ra_ca
!!!  END INTERFACE
!!!!
  INTERFACE four2D_real
     MODULE PROCEDURE four2D_ra_ca
  END INTERFACE
!
!!!  INTERFACE four1D
!!!     MODULE PROCEDURE four1D_ca
!!!  END INTERFACE
!!!!
  INTERFACE fourcol
     MODULE PROCEDURE fourcol_ca, fourcol_caa
  END INTERFACE
!
!!!  INTERFACE fourrow
!!!     MODULE PROCEDURE fourrow_ca
!!!  END INTERFACE
!
CONTAINS
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four1D_ra_ca(vec_ra, vec_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:), INTENT(INOUT) :: vec_ra
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: vec_ca
    INTEGER,               INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=1
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ra, dim1_ca, i, id, istat
    REAL,    DIMENSION(:), ALLOCATABLE :: vec_ra_tmp
    COMPLEX, DIMENSION(:), ALLOCATABLE :: vec_ca_tmp
!
!
    dim1_ra = SIZE(vec_ra)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
           WRITE(*,*) 'FOUR1D_RA_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(vec_ra_tmp(dim1_ra), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: Allocation of  vec_ra_tmp  failed!'
          STOP
       END IF
!
       dim1_ca = SIZE(vec_ca)
       ALLOCATE(vec_ca_tmp(dim1_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: Allocation of  vec_ca_tmp  failed!'
          STOP
       END IF
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
#if defined(real_precision_default)
          CALL sfftw_plan_dft_r2c_1d(plan1d(id,NUM), dim1_ra, vec_ra_tmp(1), &
               vec_ca_tmp(1), FFTW_ESTIMATE)
#else
          CALL dfftw_plan_dft_r2c_1d(plan1d(id,NUM), dim1_ra, vec_ra_tmp(1), &
               vec_ca_tmp(1), FFTW_ESTIMATE)
#endif
       CASE (1)
#if defined(real_precision_default)
          CALL sfftw_plan_dft_c2r_1d(plan1d(id,NUM), dim1_ra, vec_ca_tmp(1), &
               vec_ra_tmp(1), FFTW_ESTIMATE)
#else
          CALL dfftw_plan_dft_c2r_1d(plan1d(id,NUM), dim1_ra, vec_ca_tmp(1), &
               vec_ra_tmp(1), FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(vec_ra_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: Dellocation of  vec_ra_tmp  failed!'
          STOP
       END IF
!
       DEALLOCATE(vec_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: Dellocation of  vec_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
! Using the Guru execution of plans.
!
    SELECT CASE (isign)
    CASE (-1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_r2c(plan1d(id,NUM), vec_ra(1), vec_ca(1))
#else
       CALL dfftw_execute_dft_r2c(plan1d(id,NUM), vec_ra(1), vec_ca(1))
#endif
    CASE (1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_c2r(plan1d(id,NUM), vec_ca(1), vec_ra(1))
#else
       CALL dfftw_execute_dft_c2r(plan1d(id,NUM), vec_ca(1), vec_ra(1))
#endif
    END SELECT
!
  END SUBROUTINE four1D_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=2, RANK=1 
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ra, dim2_ra, dim1_ca, dim2_ca
    INTEGER :: idist, odist, howmany, i, id, istat
    INTEGER, DIMENSION(RANK)             :: n_arr, inembed, onembed
    REAL,    DIMENSION(:,:), ALLOCATABLE :: arr_ra_tmp
    COMPLEX, DIMENSION(:,:), ALLOCATABLE :: arr_ca_tmp
!
    dim1_ra = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
    howmany = dim2_ra
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_ra_tmp(dim1_ra, dim2_ra), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Allocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       dim1_ca = SIZE(arr_ca,1)
       dim2_ca = SIZE(arr_ca,2)
       ALLOCATE(arr_ca_tmp(dim1_ca, dim2_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Allocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
          n_arr(1) = dim1_ra
          howmany  = dim2_ra
          inembed(1) = SIZE(arr_ra)
          onembed(1) = SIZE(arr_ca)
          idist = dim1_ra
          odist = dim1_ca
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft_r2c(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ra_tmp(1,1), inembed, 1, idist, &
               arr_ca_tmp(1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft_r2c(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ra_tmp(1,1), inembed, 1, idist, &
               arr_ca_tmp(1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#endif
       CASE (1)
          n_arr(1) = dim1_ra
          howmany  = dim2_ca
          inembed(1) = SIZE(arr_ca)
          onembed(1) = SIZE(arr_ra)
          idist = dim1_ca
          odist = dim1_ra
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft_c2r(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), inembed, 1, idist, &
               arr_ra_tmp(1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft_c2r(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), inembed, 1, idist, &
               arr_ra_tmp(1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(arr_ra_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Dellocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       DEALLOCATE(arr_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Dellocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
! Using the Guru execution of plans.
!
    SELECT CASE (isign)
    CASE (-1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_r2c(plan1d(id,NUM), arr_ra(1,1), arr_ca(1,1))
#else
       CALL dfftw_execute_dft_r2c(plan1d(id,NUM), arr_ra(1,1), arr_ca(1,1))
#endif
    CASE (1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_c2r(plan1d(id,NUM), arr_ca(1,1), arr_ra(1,1))
#else
       CALL dfftw_execute_dft_c2r(plan1d(id,NUM), arr_ca(1,1), arr_ra(1,1))
#endif
    END SELECT
!
  END SUBROUTINE fourcol_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_raa_caa(arr_raa, arr_caa, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:,:), INTENT(INOUT) :: arr_raa
    COMPLEX, DIMENSION(:,:,:), INTENT(INOUT) :: arr_caa
    INTEGER,                   INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=3, RANK=1
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_raa, dim2_raa, dim3_raa, dim1_caa, dim2_caa, dim3_caa
    INTEGER :: idist, odist, howmany, i, id, istat
    INTEGER, DIMENSION(RANK)               :: n_arr, inembed, onembed
    REAL,    DIMENSION(:,:,:), ALLOCATABLE :: arr_raa_tmp
    COMPLEX, DIMENSION(:,:,:), ALLOCATABLE :: arr_caa_tmp
!
    dim1_raa = SIZE(arr_raa,1)
    dim2_raa = SIZE(arr_raa,2)
    dim3_raa = SIZE(arr_raa,3)
    howmany = dim2_raa*dim3_raa
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_raa
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_raa_tmp(dim1_raa, dim2_raa, dim3_raa), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: Allocation of  arr_raa_tmp  failed!'
          STOP
       END IF
!
       dim1_caa = SIZE(arr_caa,1)
       dim2_caa = SIZE(arr_caa,2)
       dim3_caa = SIZE(arr_caa,3)
       ALLOCATE(arr_caa_tmp(dim1_caa, dim2_caa, dim3_caa), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: Allocation of  arr_caa_tmp  failed!'
          STOP
       END IF
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
          n_arr(1) = dim1_raa
          howmany  = dim2_raa*dim3_raa
          inembed(1) = SIZE(arr_raa)
          onembed(1) = SIZE(arr_caa)
          idist = dim1_raa
          odist = dim1_caa
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft_r2c(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_raa_tmp(1,1,1), inembed, 1, idist, &
               arr_caa_tmp(1,1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft_r2c(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_raa_tmp(1,1,1), inembed, 1, idist, &
               arr_caa_tmp(1,1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#endif
       CASE (1)
          n_arr(1) = dim1_raa
          howmany  = dim2_caa*dim3_caa
          inembed(1) = SIZE(arr_caa)
          onembed(1) = SIZE(arr_raa)
          idist = dim1_caa
          odist = dim1_raa
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft_c2r(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), inembed, 1, idist, &
               arr_raa_tmp(1,1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft_c2r(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), inembed, 1, idist, &
               arr_raa_tmp(1,1,1), onembed, 1, odist, &
               FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(arr_raa_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: Dellocation of  arr_raa_tmp  failed!'
          STOP
       END IF
!
       DEALLOCATE(arr_caa_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: Dellocation of  arr_caa_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
! Using the Guru execution of plans.
!
    SELECT CASE (isign)
    CASE (-1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_r2c(plan1d(id,NUM), arr_raa(1,1,1), arr_caa(1,1,1))
#else
       CALL dfftw_execute_dft_r2c(plan1d(id,NUM), arr_raa(1,1,1), arr_caa(1,1,1))
#endif
    CASE (1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_c2r(plan1d(id,NUM), arr_caa(1,1,1), arr_raa(1,1,1))
#else
       CALL dfftw_execute_dft_c2r(plan1d(id,NUM), arr_caa(1,1,1), arr_raa(1,1,1))
#endif
    END SELECT
!
  END SUBROUTINE fourcol_raa_caa
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
#if defined(INEEDTHISTOWORK)

  SUBROUTINE fourrow_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=4, RANK=1 
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ra, dim2_ra, dim1_ca, dim2_ca
    INTEGER :: idist, odist, howmany, i, id, istat
    INTEGER, DIMENSION(RANK)             :: n_arr, inembed, onembed
    REAL,    DIMENSION(:,:), ALLOCATABLE :: arr_ra_tmp
    COMPLEX, DIMENSION(:,:), ALLOCATABLE :: arr_ca_tmp
!
    dim1_ra = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
    howmany = dim1_ra
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim2_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_ra_tmp(dim1_ra, dim2_ra), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Allocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       dim1_ca = SIZE(arr_ca,1)
       dim2_ca = SIZE(arr_ca,2)
       ALLOCATE(arr_ca_tmp(dim1_ca, dim2_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Allocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
          n_arr(1) = dim2_ra
          howmany  = dim1_ra
          inembed(1) = SIZE(arr_ra)
          onembed(1) = SIZE(arr_ca)
          CALL dfftw_plan_many_dft_r2c(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ra_tmp(1,1), inembed, howmany, 1, &
               arr_ca_tmp(1,1), onembed, howmany, 1, &
               FFTW_ESTIMATE)
       CASE (1)
          n_arr(1) = dim2_ra
          howmany  = dim1_ca
          inembed(1) = SIZE(arr_ca)
          onembed(1) = SIZE(arr_ra)
          CALL dfftw_plan_many_dft_c2r(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), inembed, howmany, 1,&
               arr_ra_tmp(1,1), onembed, howmany, 1, &
               FFTW_ESTIMATE)
       END SELECT
!
       DEALLOCATE(arr_ra_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Dellocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       DEALLOCATE(arr_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: Dellocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
! Using the Guru execution of plans.
!
    SELECT CASE (isign)
    CASE (-1)
       CALL dfftw_execute_dft_r2c(plan1d(id,NUM), arr_ra(1,1), arr_ca(1,1))
    CASE (1)
       CALL dfftw_execute_dft_c2r(plan1d(id,NUM), arr_ca(1,1), arr_ra(1,1))
    END SELECT
!
  END SUBROUTINE fourrow_ra_ca

#endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four2D_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=1
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ra, dim2_ra, dim1_ca, dim2_ca, i, id, istat
    REAL,    DIMENSION(:,:), ALLOCATABLE :: arr_ra_tmp
    COMPLEX, DIMENSION(:,:), ALLOCATABLE :: arr_ca_tmp
!
!
    dim1_ra = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n2d_saved(NUM)
       IF (k == n2d_par(i,NUM)%par(1) .AND. &
            dim2_ra == n2d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n2d_saved(NUM) == MXPLAN) THEN
           WRITE(*,*) 'FOUR2D_RA_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_ra_tmp(dim1_ra, dim2_ra), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: Allocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       dim1_ca = SIZE(arr_ca,1)
       dim2_ca = SIZE(arr_ca,2)
       ALLOCATE(arr_ca_tmp(dim1_ca, dim2_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: Allocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
       n2d_saved(NUM) = n2d_saved(NUM)+1
       n2d_par(n2d_saved(NUM),NUM)%par(1) = k
       n2d_par(n2d_saved(NUM),NUM)%par(2) = dim2_ra
       id = n2d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
#if defined(real_precision_default)
          CALL sfftw_plan_dft_r2c_2d(plan2d(id,NUM), dim1_ra, dim2_ra, &
               arr_ra_tmp(1,1), arr_ca_tmp(1,1), FFTW_ESTIMATE)
#else
          CALL dfftw_plan_dft_r2c_2d(plan2d(id,NUM), dim1_ra, dim2_ra, &
               arr_ra_tmp(1,1), arr_ca_tmp(1,1), FFTW_ESTIMATE)
#endif
       CASE (1)
#if defined(real_precision_default)
          CALL sfftw_plan_dft_c2r_2d(plan2d(id,NUM), dim1_ra, dim2_ra, &
               arr_ca_tmp(1,1), arr_ra_tmp(1,1), FFTW_ESTIMATE)
#else
          CALL dfftw_plan_dft_c2r_2d(plan2d(id,NUM), dim1_ra, dim2_ra, &
               arr_ca_tmp(1,1), arr_ra_tmp(1,1), FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(arr_ra_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: Dellocation of  arr_ra_tmp  failed!'
          STOP
       END IF
!
       DEALLOCATE(arr_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: Dellocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
! Using the Guru execution of plans.
!
    SELECT CASE (isign)
    CASE (-1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_r2c(plan2d(id,NUM), arr_ra(1,1), arr_ca(1,1))
#else
       CALL dfftw_execute_dft_r2c(plan2d(id,NUM), arr_ra(1,1), arr_ca(1,1))
#endif
    CASE (1)
#if defined(real_precision_default)
       CALL sfftw_execute_dft_c2r(plan2d(id,NUM), arr_ca(1,1), arr_ra(1,1))
#else
       CALL dfftw_execute_dft_c2r(plan2d(id,NUM), arr_ca(1,1), arr_ra(1,1))
#endif
    END SELECT
!
  END SUBROUTINE four2D_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
#if defined(INEEDTHISTOWORK)
  SUBROUTINE four1D_ca(vec_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: vec_ca
    INTEGER,               INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=5
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ca, i, id, istat
    COMPLEX, DIMENSION(:), ALLOCATABLE :: vec_ca_tmp
!
!
    dim1_ca = SIZE(vec_ca)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
           WRITE(*,*) 'FOUR1D_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(vec_ca_tmp(dim1_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: Allocation of  vec_ca_tmp  failed!'
          STOP
       END IF
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
          CALL dfftw_plan_dft_1d(plan1d(id,NUM), dim1_ca, &
               vec_ca_tmp(1), vec_ca_tmp(1), &
               FFTW_FORWARD, FFTW_ESTIMATE)
       CASE (1)
          CALL dfftw_plan_dft_1d(plan1d(id,NUM), dim1_ca, &
               vec_ca_tmp(1), vec_ca_tmp(1), &
               FFTW_BACKWARD, FFTW_ESTIMATE)
       END SELECT
!
       DEALLOCATE(vec_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: Dellocation of  vec_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
    CALL dfftw_execute_dft(plan1d(id,NUM), vec_ca(1), vec_ca(1))
!
  END SUBROUTINE four1D_ca

#endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_ca(arr_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=6, RANK=1 
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ca, dim2_ca, howmany, i, id, istat, n
    INTEGER, DIMENSION(RANK)             :: n_arr, nembed
    COMPLEX, DIMENSION(:,:), ALLOCATABLE :: arr_ca_tmp
!
    dim1_ca = SIZE(arr_ca,1)
    dim2_ca = SIZE(arr_ca,2)
    howmany = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURCOL_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_ca_tmp(dim1_ca, dim2_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: Allocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
       nembed(1) = SIZE(arr_ca)
       n_arr(1)  = dim1_ca
       n         = n_arr(1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               FFTW_FORWARD, FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               FFTW_FORWARD, FFTW_ESTIMATE)
#endif
       CASE (1)
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               FFTW_BACKWARD, FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               arr_ca_tmp(1,1), nembed, 1, n, &
               FFTW_BACKWARD, FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(arr_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: Dellocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
#if defined(real_precision_default)
    CALL sfftw_execute_dft(plan1d(id,NUM), arr_ca(1,1), arr_ca(1,1))
#else
    CALL dfftw_execute_dft(plan1d(id,NUM), arr_ca(1,1), arr_ca(1,1))
#endif
!
  END SUBROUTINE fourcol_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_caa(arr_caa, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:,:), INTENT(INOUT) :: arr_caa
    INTEGER,                   INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=7, RANK=1
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_caa, dim2_caa, dim3_caa, howmany, i, id, istat, n
    INTEGER, DIMENSION(RANK)               :: n_arr, nembed
    COMPLEX, DIMENSION(:,:,:), ALLOCATABLE :: arr_caa_tmp
!
    dim1_caa = SIZE(arr_caa,1)
    dim2_caa = SIZE(arr_caa,2)
    dim3_caa = SIZE(arr_caa,3)
    howmany = dim2_caa*dim3_caa
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_caa
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURCOL_CAA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_caa_tmp(dim1_caa, dim2_caa, dim3_caa), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: Allocation of  arr_caa_tmp  failed!'
          STOP
       END IF
!
       nembed(1) = SIZE(arr_caa)
       n_arr(1)  = dim1_caa
       n         = n_arr(1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               FFTW_FORWARD, FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               FFTW_FORWARD, FFTW_ESTIMATE)
#endif
       CASE (1)
#if defined(real_precision_default)
          CALL sfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               FFTW_BACKWARD, FFTW_ESTIMATE)
#else
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               arr_caa_tmp(1,1,1), nembed, 1, n, &
               FFTW_BACKWARD, FFTW_ESTIMATE)
#endif
       END SELECT
!
       DEALLOCATE(arr_caa_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: Dellocation of  arr_caa_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!

#if defined(real_precision_default)
    CALL sfftw_execute_dft(plan1d(id,NUM), arr_caa(1,1,1), arr_caa(1,1,1))
#else
    CALL dfftw_execute_dft(plan1d(id,NUM), arr_caa(1,1,1), arr_caa(1,1,1))
#endif
!
  END SUBROUTINE fourcol_caa
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
#if defined(I_NEED_THIS_TO_WORK)
  SUBROUTINE fourrow_ca(arr_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
! Local parameters
!
    INTEGER, PARAMETER :: NUM=8, RANK=1
!
! Local variables
!
    INTEGER :: k
    INTEGER :: dim1_ca, dim2_ca, howmany, i, id, istat, n
    INTEGER, DIMENSION(RANK)             :: n_arr, nembed
    COMPLEX, DIMENSION(:,:), ALLOCATABLE :: arr_ca_tmp
!
    dim1_ca = SIZE(arr_ca,1)
    dim2_ca = SIZE(arr_ca,2)
    howmany = dim1_ca
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim2_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
! test if the maximal number of plans is alredy reached.
!
       IF (n1d_saved(NUM) == MXPLAN) THEN
          WRITE(*,*) 'FOURROW_CA: MXPLAN too small! Increase it and recompile'
          STOP
       END IF
!
       ALLOCATE(arr_ca_tmp(dim1_ca, dim2_ca), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURROW_CA: Allocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
       nembed(1) = SIZE(arr_ca)
       n_arr(1)  = SIZE(arr_ca,2)
       howmany   = SIZE(arr_ca,1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       SELECT CASE (isign)
       CASE (-1)
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, howmany, 1, &
               arr_ca_tmp(1,1), nembed, howmany, 1, &
               FFTW_FORWARD, FFTW_ESTIMATE)
       CASE (1)
          CALL dfftw_plan_many_dft(plan1d(id,NUM), RANK, n_arr, howmany, &
               arr_ca_tmp(1,1), nembed, howmany, 1, &
               arr_ca_tmp(1,1), nembed, howmany, 1, &
               FFTW_BACKWARD, FFTW_ESTIMATE)
       END SELECT
!
       DEALLOCATE(arr_ca_tmp, stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: Dellocation of  arr_ca_tmp  failed!'
          STOP
       END IF
!
    END SELECT
!
    CALL dfftw_execute_dft(plan1d(id,NUM), arr_ca(1,1), arr_ca(1,1))
!
  END SUBROUTINE fourrow_ca
!
#endif



END MODULE fft
#endif
!
#if defined(fft_essl)
MODULE fft
!
  IMPLICIT NONE
!
  PRIVATE
  PUBLIC :: four1D_real, fourcol_real, fourrow_real
  PUBLIC :: four2D_real
  PUBLIC :: four1D, fourcol, fourrow
!
!-----------------------------------------------
!   G l o b a l   V a r i a b l e s
!-----------------------------------------------
!
! initialization of the module
!
  LOGICAL, SAVE :: initflag=.TRUE.
!
! string to copy error list entry
!
  CHARACTER (len=8), SAVE :: S2015
!
  EXTERNAL :: ENOTRM
!
! auxilary arrays
!
  REAL, DIMENSION(15) :: aux1
  REAL, DIMENSION(1)  :: aux2
!
! define the maximum number of work arrays.
!
  INTEGER, PARAMETER :: MXPLAN=16
!
! work arrays for the ESSL routine
!
  TYPE pointer_ra
     REAL, DIMENSION(:), POINTER :: poi_ra
  END TYPE pointer_ra
!
  TYPE(pointer_ra), DIMENSION(:,:), ALLOCATABLE, SAVE :: aux1d_poi1, aux1d_poi2
  TYPE(pointer_ra), DIMENSION(:,:), ALLOCATABLE, SAVE :: aux2d_poi1, aux2d_poi2
!
! size of transform
!
  TYPE int_para
     INTEGER, DIMENSION(2) :: par
  END TYPE int_para
!
  TYPE(int_para), DIMENSION(MXPLAN,8), SAVE :: n1d_par
  TYPE(int_para), DIMENSION(MXPLAN,1), SAVE :: n2d_par
!
! number of plans saved
!
  INTEGER, DIMENSION(8), SAVE :: n1d_saved=0
  INTEGER, DIMENSION(1), SAVE :: n2d_saved=0
!
  INTERFACE four1D_real
     MODULE PROCEDURE four1D_ra_ca
  END INTERFACE
!
  INTERFACE fourcol_real
     MODULE PROCEDURE fourcol_ra_ca, fourcol_raa_caa
  END INTERFACE
!
  INTERFACE fourrow_real
     MODULE PROCEDURE fourrow_ra_ca
  END INTERFACE
!
  INTERFACE four2D_real
     MODULE PROCEDURE four2D_ra_ca
  END INTERFACE
!
  INTERFACE four1D
     MODULE PROCEDURE four1D_ca
  END INTERFACE
!
  INTERFACE fourcol
     MODULE PROCEDURE fourcol_ca, fourcol_caa
  END INTERFACE
!
  INTERFACE fourrow
     MODULE PROCEDURE fourrow_ca
  END INTERFACE
!
CONTAINS
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four2d_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=1
    INTEGER :: dim1_ra, dim2_ra, dim1_ca, dim2_ca, n1_ra, i, id, istat, &
         k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux2d_poi1(MXPLAN,1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: 1. Allocation of  aux2d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux2d_poi2(MXPLAN,1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: 1. Allocation of  aux2d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ra = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
    dim1_ca = SIZE(arr_ca,1)
    dim2_ca = SIZE(arr_ca,2)
!
! the number of rows of data - that is, the length of the columns in
! array arr_ra involved in the computation.
!
    n1_ra = (dim1_ca-1)*2
!
    IF (dim1_ra < n1_ra+2) THEN
       WRITE(*,*) 'FOUR2D_RA_CA: The stride of the array arr_ra is too small!'
       STOP
    END IF
!
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n2d_saved(NUM)
       IF (k == n2d_par(i,NUM)%par(1) .AND. &
            dim2_ra == n2d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n2d_saved(NUM) = n2d_saved(NUM)+1
       n2d_par(n2d_saved(NUM),NUM)%par(1) = k
       n2d_par(n2d_saved(NUM),NUM)%par(2) = dim2_ra
       id = n2d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux1 = 1
       naux2 = SIZE(aux2)
!
       SELECT CASE (isign)
       CASE(-1)
          CALL drcft2(1, arr_ra(1,1), dim1_ra,  &
               arr_ca(1,1), dim1_ca, &
               n1_ra, dim2_ra, -isign, 1.0, aux1, naux1, aux2, naux2)
       CASE(1)
          CALL dcrft2(1, arr_ca(1,1), dim1_ca,  &
               arr_ra(1,1), dim1_ra, &
               n1_ra, dim2_ra, -isign, 1.0, aux1, naux1, aux2, naux2)
       END SELECT
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux2d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: 2. Allocation of  aux2d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux2d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR2D_RA_CA: 2. Allocation of  aux2d_poi2  failed!'
          STOP
       ENDIF
!
       SELECT CASE (isign)
       CASE(-1)
          CALL drcft2(1, arr_ra(1,1), dim1_ra, &
               arr_ca(1,1), dim1_ca, &
               n1_ra, dim2_ra, -isign, 1.0, &
               aux2d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux2d_poi2(id,NUM)%poi_ra(1), naux2)
       CASE(1)
          CALL dcrft2(1, arr_ca(1,1), dim1_ca, &
               arr_ra(1,1), dim1_ra, &
               n1_ra, dim2_ra, -isign, 1.0, &
               aux2d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux2d_poi2(id,NUM)%poi_ra(1), naux2)
       END SELECT
!
    END SELECT
!
    SELECT CASE (isign)
    CASE(-1)
       CALL drcft2(0, arr_ra(1,1), dim1_ra, &
            arr_ca(1,1), dim1_ca, &
            n1_ra, dim2_ra, -isign, 1.0,  &
            aux2d_poi1(id,NUM)%poi_ra(1), SIZE(aux2d_poi1(id,NUM)%poi_ra), &
            aux2d_poi2(id,NUM)%poi_ra(1), SIZE(aux2d_poi2(id,NUM)%poi_ra))
    CASE(1)
       CALL dcrft2(0, arr_ca(1,1), dim1_ca, &
            arr_ra(1,1), dim1_ra, &
            n1_ra, dim2_ra, -isign, 1.0,  &
            aux2d_poi1(id,NUM)%poi_ra(1), SIZE(aux2d_poi1(id,NUM)%poi_ra), &
            aux2d_poi2(id,NUM)%poi_ra(1), SIZE(aux2d_poi2(id,NUM)%poi_ra))
    END SELECT
!
  END SUBROUTINE four2d_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four1D_ra_ca(vec_ra, vec_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:), INTENT(inout) :: vec_ra
    COMPLEX, DIMENSION(:), INTENT(inout) :: vec_ca
    INTEGER,               INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=1
    INTEGER :: dim1_ra, dim1_ca, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ra = SIZE(vec_ra)
    dim1_ca = SIZE(vec_ca)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux2 = SIZE(aux2)
!
       SELECT CASE (isign)
       CASE(-1)
          naux1 = 15
          CALL drcft(1, vec_ra(1), dim1_ra,  &
               vec_ca(1), dim1_ca, &
               dim1_ra, 1, -isign, 1.0, aux1, naux1, aux2, naux2)
       CASE(1)
          naux1 = 14
          CALL dcrft(1, vec_ca(1), dim1_ca,  &
               vec_ra(1), dim1_ra, &
               dim1_ra, 1, -isign, 1.0, aux1, naux1, aux2, naux2)
       END SELECT
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_RA_CA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       SELECT CASE (isign)
       CASE(-1)
          CALL drcft(1, vec_ra(1), dim1_ra, &
               vec_ca(1), dim1_ca, &
               dim1_ra, 1, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       CASE(1)
          CALL dcrft(1, vec_ca(1), dim1_ca, &
               vec_ra(1), dim1_ra, &
               dim1_ra, 1, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       END SELECT
!
    END SELECT
!
    SELECT CASE (isign)
    CASE(-1)
       CALL drcft(0, vec_ra(1), dim1_ra, &
            vec_ca(1), dim1_ca, &
            dim1_ra, 1, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    CASE(1)
       CALL dcrft(0, vec_ca(1), dim1_ca, &
            vec_ra(1), dim1_ra, &
            dim1_ra, 1, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    END SELECT
!
  END SUBROUTINE four1D_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=2
    INTEGER :: dim1_ra, dim1_ca, howmany, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ra = SIZE(arr_ra,1)
    howmany = SIZE(arr_ra,2)
    dim1_ca = SIZE(arr_ca,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux2 = SIZE(aux2)
!
       SELECT CASE (isign)
       CASE(-1)
          naux1 = 15
          CALL drcft(1, arr_ra(1,1), dim1_ra,  &
               arr_ca(1,1), dim1_ca, &
               dim1_ra, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
       CASE(1)
          naux1 = 14
          CALL dcrft(1, arr_ca(1,1), dim1_ca,  &
               arr_ra(1,1), dim1_ra, &
               dim1_ra, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
       END SELECT
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RA_CA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       SELECT CASE (isign)
       CASE(-1)
          CALL drcft(1, arr_ra(1,1), dim1_ra, &
               arr_ca(1,1), dim1_ca, &
               dim1_ra, howmany, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       CASE(1)
          CALL dcrft(1, arr_ca(1,1), dim1_ca, &
               arr_ra(1,1), dim1_ra, &
               dim1_ra, howmany, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       END SELECT
!
    END SELECT
!
    SELECT CASE (isign)
    CASE(-1)
       CALL drcft(0, arr_ra(1,1), dim1_ra, &
            arr_ca(1,1), dim1_ca, &
            dim1_ra, howmany, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    CASE(1)
       CALL dcrft(0, arr_ca(1,1), dim1_ca, &
            arr_ra(1,1), dim1_ra, &
            dim1_ra, howmany, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    END SELECT
!
  END SUBROUTINE fourcol_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_raa_caa(arr_raa, arr_caa, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:,:), INTENT(inout) :: arr_raa
    COMPLEX, DIMENSION(:,:,:), INTENT(inout) :: arr_caa
    INTEGER,                   INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=3
    INTEGER :: dim1_raa, dim1_caa, howmany, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_raa = SIZE(arr_raa,1)
    howmany  = SIZE(arr_raa,2)*SIZE(arr_raa,3)
    dim1_caa = SIZE(arr_caa,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_raa
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux2 = SIZE(aux2)
!
       SELECT CASE (isign)
       CASE(-1)
          naux1 = 15
          CALL drcft(1, arr_raa(1,1,1), dim1_raa,  &
               arr_caa(1,1,1), dim1_caa, &
               dim1_raa, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
       CASE(1)
          naux1 = 14
          CALL dcrft(1, arr_caa(1,1,1), dim1_caa,  &
               arr_raa(1,1,1), dim1_raa, &
               dim1_raa, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
       END SELECT
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_RAA_CAA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       SELECT CASE (isign)
       CASE(-1)
          CALL drcft(1, arr_raa(1,1,1), dim1_raa, &
               arr_caa(1,1,1), dim1_caa, &
               dim1_raa, howmany, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       CASE(1)
          CALL dcrft(1, arr_caa(1,1,1), dim1_caa, &
               arr_raa(1,1,1), dim1_raa, &
               dim1_raa, howmany, -isign, 1.0, &
               aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
               aux1d_poi2(id,NUM)%poi_ra(1), naux2)
       END SELECT
!
    END SELECT
!
    SELECT CASE (isign)
    CASE(-1)
       CALL drcft(0, arr_raa(1,1,1), dim1_raa, &
            arr_caa(1,1,1), dim1_caa, &
            dim1_raa, howmany, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    CASE(1)
       CALL dcrft(0, arr_caa(1,1,1), dim1_caa, &
            arr_raa(1,1,1), dim1_raa, &
            dim1_raa, howmany, -isign, 1.0,  &
            aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
            aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
    END SELECT
!
  END SUBROUTINE fourcol_raa_caa
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourrow_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!
    SELECT CASE (isign)
    CASE(-1)
       WRITE(*,*) 'FOURROW_RA_CA: Real-to-Complex FFT of rows'// &
            ' not implementable in ESSL v4.2'
    CASE(1)
       WRITE(*,*) 'FOURROW_RA_CA: Complex-to-Real FFT of rows'// &
            ' not implementable in ESSL v4.2'
    END SELECT
!
    STOP
!
  END SUBROUTINE fourrow_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four1D_ca(vec_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:), INTENT(inout) :: vec_ca
    INTEGER,               INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=5
    INTEGER :: dim1_ca, i, id, istat, k, n, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ca = SIZE(vec_ca)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux1 = 8
       naux2 = SIZE(aux2)
!
       CALL dcft(1, vec_ca(1), 1, dim1_ca,  vec_ca(1), 1, dim1_ca, &
            dim1_ca, 1, -isign, 1.0, aux1, naux1, aux2, naux2)
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOUR1D_CA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       CALL dcft(1, vec_ca(1), 1, dim1_ca, vec_ca(1), 1, dim1_ca, &
            dim1_ca, 1, -isign, 1.0, &
            aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
            aux1d_poi2(id,NUM)%poi_ra(1), naux2)
!
    END SELECT
!
    CALL dcft(0, vec_ca(1), 1, dim1_ca, vec_ca(1), 1, dim1_ca, &
         dim1_ca, 1, -isign, 1.0,  &
         aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
         aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
!
  END SUBROUTINE four1D_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_ca(arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=6
    INTEGER :: dim1_ca, howmany, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,8), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ca = SIZE(arr_ca,1)
    howmany = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux1 = 8
       naux2 = SIZE(aux2)
!
       CALL dcft(1, arr_ca(1,1), 1, dim1_ca,  arr_ca(1,1), 1, dim1_ca, &
            dim1_ca, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       CALL dcft(1, arr_ca(1,1), 1, dim1_ca, arr_ca(1,1), 1, dim1_ca, &
            dim1_ca, howmany, -isign, 1.0, &
            aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
            aux1d_poi2(id,NUM)%poi_ra(1), naux2)
!
    END SELECT
!
    CALL dcft(0, arr_ca(1,1), 1, dim1_ca, arr_ca(1,1), 1, dim1_ca, &
         dim1_ca, howmany, -isign, 1.0, &
         aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
         aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
!
  END SUBROUTINE fourcol_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_caa(arr_caa, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:,:), INTENT(inout) :: arr_caa
    INTEGER,                   INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=7
    INTEGER :: dim1_caa, howmany, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_caa = SIZE(arr_caa,1)
    howmany  = SIZE(arr_caa,2)*SIZE(arr_caa,3)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_caa
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux1 = 8
       naux2 = SIZE(aux2)
!
       CALL dcft(1, arr_caa(1,1,1), 1, dim1_caa, arr_caa(1,1,1), 1, dim1_caa, &
            dim1_caa, howmany, -isign, 1.0, aux1, naux1, aux2, naux2)
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arrays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURCOL_CAA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       CALL dcft(1, arr_caa(1,1,1), 1, dim1_caa, arr_caa(1,1,1), 1, dim1_caa, &
            dim1_caa, howmany, -isign, 1.0, &
            aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
            aux1d_poi2(id,NUM)%poi_ra(1), naux2)
!
    END SELECT
!
    CALL dcft(0, arr_caa(1,1,1), 1, dim1_caa, arr_caa(1,1,1), 1, dim1_caa, &
         dim1_caa, howmany, -isign, 1.0, &
         aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
         aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
!
  END SUBROUTINE fourcol_caa
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourrow_ca(arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=8
    INTEGER :: dim1_ca, dim2_ca, i, id, istat, k, naux1, naux2
! -----------------------------------------------
!
    IF (initflag) THEN
       initflag = .FALSE.
!
       CALL EINFO(0)
       CALL ERRSAV(2015,S2015)
!
       ALLOCATE(aux1d_poi1(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURROW_CA: 1. Allocation of  aux1d_poi1  failed!'
          STOP
       END IF
!
       ALLOCATE(aux1d_poi2(MXPLAN,NUM), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURROW_CA: 1. Allocation of  aux1d_poi2  failed!'
          STOP
       END IF
!
    END IF
!
    dim1_ca = SIZE(arr_ca,1)
    dim2_ca = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            dim2_ca == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
!
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = dim2_ca
       id = n1d_saved(NUM)
!
       CALL ERRSET(2015,0,-1,1,ENOTRM,0)
!
       naux1 = 8
       naux2 = SIZE(aux2)
!
       CALL dcft(1, arr_ca(1,1), dim2_ca, 1,  arr_ca(1,1), dim2_ca, 1, &
            dim1_ca, dim2_ca, -isign, 1.0, aux1, naux1, aux2, naux2)
!
       CALL ERRSTR(2015,S2015)
!
! dynamic allocation of the work arr_caays.
!
       ALLOCATE(aux1d_poi1(id,NUM)%poi_ra(naux1), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURROW_CA: 2. Allocation of  aux1d_poi1  failed!'
          STOP
       ENDIF
!
       ALLOCATE(aux1d_poi2(id,NUM)%poi_ra(naux2), stat=istat)
       IF (istat /= 0) THEN
          WRITE(*,*) 'FOURROW_CA: 2. Allocation of  aux1d_poi2  failed!'
          STOP
       ENDIF
!
       CALL dcft(1, arr_ca(1,1), dim1_ca, 1, arr_ca(1,1), dim1_ca, 1, &
            dim1_ca, dim2_ca, -isign, 1.0, &
            aux1d_poi1(id,NUM)%poi_ra(1), naux1, &
            aux1d_poi2(id,NUM)%poi_ra(1), naux2)
!
    END SELECT
!
    CALL dcft(0, arr_ca(1,1), dim1_ca, 1, arr_ca(1,1), dim1_ca, 1, &
         dim1_ca, dim2_ca, -isign, 1.0,  &
         aux1d_poi1(id,NUM)%poi_ra(1), SIZE(aux1d_poi1(id,NUM)%poi_ra), &
         aux1d_poi2(id,NUM)%poi_ra(1), SIZE(aux1d_poi2(id,NUM)%poi_ra))
!
  END SUBROUTINE fourrow_ca
! 
!-----------------------------------------------------------------------
! 
END MODULE fft
#endif
!
#if defined(fft_mkl)
MODULE fft
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE mkl_dfti
!
  IMPLICIT NONE
!
  PRIVATE
  PUBLIC :: pointer_r, handle1d, handle2d
  PUBLIC :: four1D_real, fourcol_real, fourrow_real
  PUBLIC :: four2D_real
  PUBLIC :: four1D, fourcol, fourrow
!
!-----------------------------------------------
!   G l o b a l   V a r i a b l e s
!-----------------------------------------------
!
! define the maximum number of plans.
!
  INTEGER, PARAMETER :: MXPLAN=16
!
  TYPE(DFTI_DESCRIPTOR), POINTER :: desc_handle
!
  TYPE pointer_r
     TYPE(DFTI_DESCRIPTOR), POINTER :: desc_handle
  END TYPE pointer_r
!
! descriptor handles for 1-dim ans 2-dim FFT
!
  TYPE(pointer_r), DIMENSION(MXPLAN,8), SAVE :: handle1d
  TYPE(pointer_r), DIMENSION(MXPLAN,1), SAVE :: handle2d
!
! size of transform
!
  TYPE int_para
     INTEGER, DIMENSION(2) :: par
  END TYPE int_para
!
! size of transform
!
  TYPE(int_para), DIMENSION(MXPLAN,8), SAVE :: n1d_par
  TYPE(int_para), DIMENSION(MXPLAN,1), SAVE :: n2d_par
!
! number of descriptor handles saved
!
  INTEGER, DIMENSION(8), SAVE :: n1d_saved=0
  INTEGER, DIMENSION(1), SAVE :: n2d_saved=0
!
  INTERFACE four1D_real
     MODULE PROCEDURE four1D_ra_ca
  END INTERFACE
!
  INTERFACE fourcol_real
     MODULE PROCEDURE fourcol_ra_ca, fourcol_raa_caa
  END INTERFACE
!
  INTERFACE fourrow_real
     MODULE PROCEDURE fourrow_ra_ca
  END INTERFACE
!
  INTERFACE four2D_real
     MODULE PROCEDURE four2D_ra_ca
  END INTERFACE
!
  INTERFACE four1D
     MODULE PROCEDURE four1D_ca
  END INTERFACE
!
  INTERFACE fourcol
     MODULE PROCEDURE fourcol_ca, fourcol_caa
  END INTERFACE
!
  INTERFACE fourrow
     MODULE PROCEDURE fourrow_ca
  END INTERFACE
!
CONTAINS
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four2D_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=1
    LOGICAL :: init_flag
    INTEGER :: dim1_ra, dim2_ra, dim1_ca, id, i, k, status
! -----------------------------------------------
!
    dim1_ra = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
    dim1_ca = SIZE(arr_ca,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n2d_saved(NUM)
       IF (k == n2d_par(i,NUM)%par(1) .AND. &
            dim2_ra == n2d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n2d_saved(NUM) = n2d_saved(NUM)+1
       n2d_par(n2d_saved(NUM),NUM)%par(1) = k
       n2d_par(n2d_saved(NUM),NUM)%par(2) = dim2_ra
       id = n2d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL four2D_mkl_ra_ca(arr_ra(1,1), arr_ca(1,1), dim1_ra, dim2_ra, &
         dim1_ca, isign, init_flag, id, NUM)
!
  END SUBROUTINE four2D_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four1D_ra_ca(vec_ra, vec_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    INTEGER,               INTENT(in)    :: isign
    REAL,    DIMENSION(:), INTENT(INOUT) :: vec_ra
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: vec_ca
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=1
    LOGICAL :: init_flag
    INTEGER :: dim1_ra, dim1_ca, id, i, k, status
! -----------------------------------------------
!
    dim1_ra = SIZE(vec_ra,1)
    dim1_ca = SIZE(vec_ca,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ra
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ra_ca(vec_ra(1), vec_ca(1), dim1_ra, dim1_ca, &
         1, isign, init_flag, id, NUM)
!
  END SUBROUTINE four1D_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=2
    LOGICAL :: init_flag
    INTEGER :: dim1_ra, dim1_ca, howmany, id, i, k, status
! -----------------------------------------------
!
    dim1_ra = SIZE(arr_ra,1)
    howmany = SIZE(arr_ra,2)
    dim1_ca = SIZE(arr_ca,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_ra)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ra_ca(arr_ra(1,1), arr_ca(1,1), dim1_ra, dim1_ca, &
         howmany, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourcol_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_raa_caa(arr_raa, arr_caa, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:,:), INTENT(inout) :: arr_raa
    COMPLEX, DIMENSION(:,:,:), INTENT(inout) :: arr_caa
    INTEGER,                   INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=3
    LOGICAL :: init_flag
    INTEGER :: dim1_raa, dim1_caa, howmany, id, i, k, status
! -----------------------------------------------
!
    dim1_raa = SIZE(arr_raa,1)
    howmany  = SIZE(arr_raa,2)*SIZE(arr_raa,3)
    dim1_caa = SIZE(arr_caa,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_raa)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ra_ca(arr_raa(1,1,1), arr_caa(1,1,1), dim1_raa, dim1_caa, &
         howmany, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourcol_raa_caa
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourrow_ra_ca(arr_ra, arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    REAL,    DIMENSION(:,:), INTENT(inout) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=4
    LOGICAL :: init_flag
    INTEGER :: dim2_ra, dim2_ca, howmany, id, i, k, status
! -----------------------------------------------
!
    howmany = SIZE(arr_ra,1)
    dim2_ra = SIZE(arr_ra,2)
    dim2_ca = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_ra)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourrow_mkl_ra_ca(arr_ra(1,1), arr_ca(1,1), howmany, &
         dim2_ra, dim2_ca, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourrow_ra_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE four1D_ca(vec_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    INTEGER,               INTENT(in)    :: isign
    COMPLEX, DIMENSION(:), INTENT(inout) :: vec_ca
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=5
    LOGICAL :: init_flag
    INTEGER :: dim1_ca, id, i, k, status
! -----------------------------------------------
!
    dim1_ca = SIZE(vec_ca,1)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*dim1_ca
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ca(vec_ca(1), dim1_ca, 1, isign, init_flag, id, NUM)
!
  END SUBROUTINE four1D_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_ca(arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=6
    LOGICAL :: init_flag
    INTEGER :: dim_ca, howmany, id, i, k, status
! -----------------------------------------------
!
    dim_ca  = SIZE(arr_ca,1)
    howmany = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_ca)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ca(arr_ca(1,1), dim_ca, howmany, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourcol_ca
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourcol_caa(arr_caa, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:,:), INTENT(inout) :: arr_caa
    INTEGER,                   INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=7
    LOGICAL :: init_flag
    INTEGER :: dim_caa, howmany, id, i, k, status
! -----------------------------------------------
!
    dim_caa = SIZE(arr_caa,1)
    howmany = SIZE(arr_caa,2)*SIZE(arr_caa,3)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_caa)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourcol_mkl_ca(arr_caa(1,1,1), dim_caa, howmany, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourcol_caa
! 
!-----------------------------------------------------------------------
! 
  SUBROUTINE fourrow_ca(arr_ca, isign)
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
    COMPLEX, DIMENSION(:,:), INTENT(inout) :: arr_ca
    INTEGER,                 INTENT(in)    :: isign
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
    INTEGER, PARAMETER :: NUM=8
    LOGICAL :: init_flag
    INTEGER :: dim2_ca, howmany, id, i, k, status
! -----------------------------------------------
!
    howmany = SIZE(arr_ca,1)
    dim2_ca = SIZE(arr_ca,2)
!
! test if a plan that fits is already created.
!
    id = -1
    k = isign*SIZE(arr_ca)
    DO i = 1,n1d_saved(NUM)
       IF (k == n1d_par(i,NUM)%par(1) .AND. &
            howmany == n1d_par(i,NUM)%par(2)) THEN
          id = i
          EXIT
       END IF
    END DO
!
    SELECT CASE (id)
    CASE (-1)
       init_flag = .TRUE.
       n1d_saved(NUM) = n1d_saved(NUM)+1
       n1d_par(n1d_saved(NUM),NUM)%par(1) = k
       n1d_par(n1d_saved(NUM),NUM)%par(2) = howmany
       id = n1d_saved(NUM)
!
    CASE default
       init_flag = .FALSE.
    END SELECT
!
    CALL fourrow_mkl_ca(arr_ca(1,1), howmany, dim2_ca, isign, init_flag, id, NUM)
!
  END SUBROUTINE fourrow_ca
! 
!-----------------------------------------------------------------------
!
END MODULE fft
! 
!-----------------------------------------------------------------------
! 
SUBROUTINE fourcol_mkl_ra_ca(arr_ra, arr_ca, dim1_ra, dim1_ca, howmany, &
     isign, init_flag, id, num)
!
! COMMENT: This subroutine is necessary to prevent the Lahey/Fujitsu
!          compiler from making a copy of array arr_ra when passing
!          arguments.
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE fft, ONLY: handle1d
  USE mkl_dfti
!
  IMPLICIT NONE
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
  REAL,    DIMENSION(*), INTENT(inout) :: arr_ra
  COMPLEX, DIMENSION(*), INTENT(inout) :: arr_ca
  INTEGER,               INTENT(in)    :: dim1_ra
  INTEGER,               INTENT(in)    :: dim1_ca
  INTEGER,               INTENT(in)    :: howmany
  INTEGER,               INTENT(in)    :: isign
  LOGICAL,               INTENT(in)    :: init_flag
  INTEGER,               INTENT(in)    :: id
  INTEGER,               INTENT(in)    :: num
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
  INTEGER :: i, status
! -----------------------------------------------
!
  IF (init_flag) THEN
!
     status = DftiCreateDescriptor(handle1d(id,num)%desc_handle, &
          DFTI_DOUBLE, DFTI_REAL, 1, dim1_ra)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 0:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_PLACEMENT, DFTI_NOT_INPLACE)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 1:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_NUMBER_OF_TRANSFORMS, howmany)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 2:', DftiErrorMessage(status)
!
     SELECT CASE (isign)
     CASE (-1)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_INPUT_DISTANCE, dim1_ra)
        IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 3:', DftiErrorMessage(status)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_OUTPUT_DISTANCE, dim1_ca)
        IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 4:', DftiErrorMessage(status)
     CASE(1)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_INPUT_DISTANCE, dim1_ca)
        IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 5:', DftiErrorMessage(status)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_OUTPUT_DISTANCE, dim1_ra)
        IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 6:', DftiErrorMessage(status)
     END SELECT
!
     status = DftiCommitDescriptor(handle1d(id,num)%desc_handle)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_RA_CA 7:', DftiErrorMessage(status)
!
  END IF
!
  SELECT CASE (isign)
  CASE (-1)
     status = DftiComputeForward(handle1d(id,num)%desc_handle, arr_ra, arr_ca)
  CASE (1)
     status = DftiComputeBackward(handle1d(id,num)%desc_handle, arr_ca, arr_ra)
  END SELECT
!
END SUBROUTINE fourcol_mkl_ra_ca
! 
!-----------------------------------------------------------------------
! 
SUBROUTINE fourrow_mkl_ra_ca(arr_ra, arr_ca, howmany, dim2_ra, dim2_ca, &
     isign, init_flag, id, num)
!
! COMMENT: This subroutine is necessary to prevent the Lahey/Fujitsu
!          compiler from making a copy of array arr_ra when passing
!          arguments.
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE fft, ONLY: handle1d
  USE mkl_dfti
!
  IMPLICIT NONE
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
  REAL,    DIMENSION(*), INTENT(inout) :: arr_ra
  COMPLEX, DIMENSION(*), INTENT(inout) :: arr_ca
  INTEGER,               INTENT(in)    :: howmany
  INTEGER,               INTENT(in)    :: dim2_ra
  INTEGER,               INTENT(in)    :: dim2_ca
  INTEGER,               INTENT(in)    :: isign
  LOGICAL,               INTENT(in)    :: init_flag
  INTEGER,               INTENT(in)    :: id
  INTEGER,               INTENT(in)    :: num
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
  INTEGER :: i, status
  INTEGER, DIMENSION(2) :: stride
! -----------------------------------------------
!
  IF (init_flag) THEN
!
     stride(1) = 0
     stride(2) = howmany
!
     status = DftiCreateDescriptor(handle1d(id,num)%desc_handle, &
          DFTI_DOUBLE, DFTI_REAL, 1, dim2_ra)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 0:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_PLACEMENT, DFTI_NOT_INPLACE)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 1:', DftiErrorMessage(status)
!
     SELECT CASE (isign)
     CASE (-1)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_NUMBER_OF_TRANSFORMS, dim2_ra)
        IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 2:', DftiErrorMessage(status)
     CASE (1)
        status = DftiSetValue(handle1d(id,num)%desc_handle, &
             DFTI_NUMBER_OF_TRANSFORMS, dim2_ca)
        IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 3:', DftiErrorMessage(status)
     END SELECT
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_INPUT_DISTANCE, 1)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 4:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_INPUT_STRIDES, stride)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 5:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_OUTPUT_DISTANCE, 1)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 6:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_OUTPUT_STRIDES, stride)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 7:', DftiErrorMessage(status)
!
     status = DftiCommitDescriptor(handle1d(id,num)%desc_handle)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_RA_CA 8:', DftiErrorMessage(status)
!
  END IF
!
  SELECT CASE (isign)
  CASE (-1)
     status = DftiComputeForward(handle1d(id,num)%desc_handle, arr_ra, arr_ca)
  CASE (1)
     status = DftiComputeBackward(handle1d(id,num)%desc_handle, arr_ca, arr_ra)
  END SELECT
!
END SUBROUTINE fourrow_mkl_ra_ca
!
!-----------------------------------------------------------------------
! 
SUBROUTINE four2D_mkl_ra_ca(arr_ra, arr_ca, dim1_ra, dim2_ra, dim1_ca, &
     isign, init_flag, id, num)
!
! COMMENT: This subroutine is necessary to prevent the Lahey/Fujitsu
!          compiler from making a copy of array arr_ra when passing
!          arguments.
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE fft, ONLY: handle2d
  USE mkl_dfti
!
  IMPLICIT NONE
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
  REAL,    DIMENSION(*), INTENT(inout) :: arr_ra
  COMPLEX, DIMENSION(*), INTENT(inout) :: arr_ca
  INTEGER,               INTENT(in)    :: dim1_ra
  INTEGER,               INTENT(in)    :: dim2_ra
  INTEGER,               INTENT(in)    :: dim1_ca
  INTEGER,               INTENT(in)    :: isign
  LOGICAL,               INTENT(in)    :: init_flag
  INTEGER,               INTENT(in)    :: id
  INTEGER,               INTENT(in)    :: num
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
  INTEGER :: i, status
  INTEGER, DIMENSION(2) :: length
  INTEGER, DIMENSION(3) :: strides_in, strides_out
! -----------------------------------------------
!
  IF (init_flag) THEN
!
     length(1) = dim1_ra
     length(2) = dim2_ra
     status = DftiCreateDescriptor(handle2d(id,num)%desc_handle, &
          DFTI_DOUBLE, DFTI_REAL, 2, length)
     IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 0:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle2d(id,num)%desc_handle, &
          DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
     IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 1:', DftiErrorMessage(status)
     status = DftiSetValue(handle2d(id,num)%desc_handle, &
          DFTI_PLACEMENT, DFTI_NOT_INPLACE)
     IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 2:', DftiErrorMessage(status)
!
     SELECT CASE (isign)
     CASE (-1)
        strides_out(1) = 0
        strides_out(2) = 1
        strides_out(3) = dim1_ca
        status = DftiSetValue(handle2d(id,num)%desc_handle, &
             DFTI_OUTPUT_STRIDES, strides_out)
        IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 3:', DftiErrorMessage(status)
! 
     CASE (1)
        strides_in(1) = 0
        strides_in(2) = 1
        strides_in(3) = dim1_ca
        status = DftiSetValue(handle2d(id,num)%desc_handle, &
             DFTI_INPUT_STRIDES, strides_in)
        IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 4:', DftiErrorMessage(status)
     END SELECT
!
     status = DftiCommitDescriptor(handle2d(id,num)%desc_handle)
     IF (status /= 0) WRITE(*,*) 'FOUR2D_MKL_RA_CA 5:', DftiErrorMessage(status)
!
  END IF
!
  SELECT CASE (isign)
  CASE (-1)
     status = DftiComputeForward(handle2d(id,num)%desc_handle, arr_ra, arr_ca)
  CASE (1)
     status = DftiComputeBackward(handle2d(id,num)%desc_handle, arr_ca, arr_ra)
  END SELECT
!
END SUBROUTINE four2D_mkl_ra_ca
! 
!-----------------------------------------------------------------------
! 
SUBROUTINE fourcol_mkl_ca(arr_ca, dim1_ca, howmany, isign, init_flag, id, num)
!
! COMMENT: This subroutine is necessary to prevent the Lahey/Fujitsu
!          compiler from making a copy of array arr_ca when passing
!          arguments.
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE fft, ONLY: handle1d
  USE mkl_dfti
!
  IMPLICIT NONE
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
  COMPLEX, DIMENSION(*), INTENT(inout) :: arr_ca
  INTEGER,               INTENT(in)    :: dim1_ca
  INTEGER,               INTENT(in)    :: howmany
  INTEGER,               INTENT(in)    :: isign
  LOGICAL,               INTENT(in)    :: init_flag
  INTEGER,               INTENT(in)    :: id
  INTEGER,               INTENT(in)    :: num
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
  INTEGER :: i, status
! -----------------------------------------------
!
  IF (init_flag) THEN
!
     status = DftiCreateDescriptor(handle1d(id,num)%desc_handle, &
          DFTI_DOUBLE, DFTI_COMPLEX, 1, dim1_ca)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_CA 0:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_NUMBER_OF_TRANSFORMS, howmany)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_CA 1:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_INPUT_DISTANCE, dim1_ca)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_CA 2:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_OUTPUT_DISTANCE, dim1_ca)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_CA 3:', DftiErrorMessage(status)
!
     status = DftiCommitDescriptor(handle1d(id,num)%desc_handle)
     IF (status /= 0) WRITE(*,*) 'FOURCOL_MKL_CA 4:', DftiErrorMessage(status)
!
  END IF
!
  SELECT CASE (isign)
  CASE (-1)
     status = DftiComputeForward(handle1d(id,num)%desc_handle, arr_ca)
  CASE (1)
     status = DftiComputeBackward(handle1d(id,num)%desc_handle, arr_ca)
  END SELECT
!
END SUBROUTINE fourcol_mkl_ca
! 
!-----------------------------------------------------------------------
! 
SUBROUTINE fourrow_mkl_ca(arr_ca, howmany, dim2_ca, isign, init_flag, id, num)
!
! COMMENT: This subroutine is necessary to prevent the Lahey/Fujitsu
!          compiler from making a copy of array arr_ca when passing
!          arguments.
!
!-----------------------------------------------
!   M o d u l e s 
!-----------------------------------------------
  USE fft, ONLY: handle1d
  USE mkl_dfti
!
  IMPLICIT NONE
!
! -----------------------------------------------
!   D u m m y   A r g u m e n t s
! -----------------------------------------------
  COMPLEX, DIMENSION(*), INTENT(inout) :: arr_ca
  INTEGER,               INTENT(in)    :: howmany
  INTEGER,               INTENT(in)    :: dim2_ca
  INTEGER,               INTENT(in)    :: isign
  LOGICAL,               INTENT(in)    :: init_flag
  INTEGER,               INTENT(in)    :: id
  INTEGER,               INTENT(in)    :: num
! -----------------------------------------------
!   L o c a l   V a r i a b l e s
! -----------------------------------------------
  INTEGER :: i, status
  INTEGER, DIMENSION(2) :: stride
! -----------------------------------------------
!
  IF (init_flag) THEN
!
     stride(1) = 0
     stride(2) = howmany
!
     status = DftiCreateDescriptor(handle1d(id,num)%desc_handle, &
          DFTI_DOUBLE, DFTI_COMPLEX, 1, dim2_ca)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 0:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_NUMBER_OF_TRANSFORMS, dim2_ca)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 1:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_INPUT_DISTANCE, 1)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 2:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_INPUT_STRIDES, stride)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 3:', DftiErrorMessage(status)
!
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_OUTPUT_DISTANCE, 1)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 4:', DftiErrorMessage(status)
     status = DftiSetValue(handle1d(id,num)%desc_handle, &
          DFTI_OUTPUT_STRIDES, stride)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 5:', DftiErrorMessage(status)
!
     status = DftiCommitDescriptor(handle1d(id,num)%desc_handle)
     IF (status /= 0) WRITE(*,*) 'FOURROW_MKL_CA 6:', DftiErrorMessage(status)
!
  END IF
!
  SELECT CASE (isign)
  CASE (-1)
     status = DftiComputeForward(handle1d(id,num)%desc_handle, arr_ca)
  CASE (1)
     status = DftiComputeBackward(handle1d(id,num)%desc_handle, arr_ca)
  END SELECT
!
END SUBROUTINE fourrow_mkl_ca
#endif

#if defined(nofft)
!
! No fft -- dummy module containing only the interfaces
! that are currently required.
! 09-dec-07/: added
!
MODULE fft
!
  IMPLICIT NONE
!
  PRIVATE
  
  !> this interface obviously provides no working FFT lib!
  LOGICAL, PUBLIC :: WORKING_FFT_LIBRARY=.false.
  
  PUBLIC :: four1D_real, fourcol_real, fourrow_real
  PUBLIC :: four2D_real
  PUBLIC :: four1D, fourcol, fourrow
!
!
  INTERFACE four1D_real
     MODULE PROCEDURE four1D_ra_ca
  END INTERFACE
!
  INTERFACE fourcol_real
     MODULE PROCEDURE fourcol_ra_ca, fourcol_raa_caa
  END INTERFACE
!
  INTERFACE fourrow_real
     MODULE PROCEDURE fourrow_ra_ca
  END INTERFACE
!
  INTERFACE four2D_real
     MODULE PROCEDURE four2D_ra_ca
  END INTERFACE
!
  INTERFACE four1D
     MODULE PROCEDURE four1D_ca
  END INTERFACE
!
  INTERFACE fourcol
     MODULE PROCEDURE fourcol_ca, fourcol_caa
  END INTERFACE
!
  INTERFACE fourrow
     MODULE PROCEDURE fourrow_ca
  END INTERFACE
!
CONTAINS
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four1D_ra_ca(vec_ra, vec_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:), INTENT(INOUT) :: vec_ra
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: vec_ca
    INTEGER,               INTENT(IN)    :: isign
!
  END SUBROUTINE four1D_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
  END SUBROUTINE fourcol_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_raa_caa(arr_raa, arr_caa, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:,:), INTENT(INOUT) :: arr_raa
    COMPLEX, DIMENSION(:,:,:), INTENT(INOUT) :: arr_caa
    INTEGER,                   INTENT(IN)    :: isign
!
  END SUBROUTINE fourcol_raa_caa
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourrow_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
  END SUBROUTINE fourrow_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four2D_ra_ca(arr_ra, arr_ca, isign)
!
! Dummy arguments
!
    REAL,    DIMENSION(:,:), INTENT(INOUT) :: arr_ra
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
  END SUBROUTINE four2D_ra_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE four1D_ca(vec_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:), INTENT(INOUT) :: vec_ca
    INTEGER,               INTENT(IN)    :: isign
!
  END SUBROUTINE four1D_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_ca(arr_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
  END SUBROUTINE fourcol_ca
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourcol_caa(arr_caa, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:,:), INTENT(INOUT) :: arr_caa
    INTEGER,                   INTENT(IN)    :: isign
!
  END SUBROUTINE fourcol_caa
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  SUBROUTINE fourrow_ca(arr_ca, isign)
!
! Dummy arguments
!
    COMPLEX, DIMENSION(:,:), INTENT(INOUT) :: arr_ca
    INTEGER,                 INTENT(IN)    :: isign
!
  END SUBROUTINE fourrow_ca
!
END MODULE fft
#endif
