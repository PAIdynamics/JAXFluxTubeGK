module genex_fortran_env_m
    !! Defines constants used in the genex
    use mpi
    use netcdf
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only : C_DOUBLE, C_FLOAT

    implicit none
    save

    integer, public, parameter :: SP = REAL32
    !! Single precision
    integer, public, parameter :: DP = REAL64
    !! Double precision

    real(kind=SP), public, parameter :: SP_EPS = epsilon(0.0_sp)
    !! Machine epsilon for single precision
    real(kind=DP), public, parameter :: DP_EPS = epsilon(0.0_dp)
    !! Machine epsilon for double precision

    real(kind=SP), public, parameter :: SP_NAN = transfer(-4194304_int32, 1._SP)
    !! NaN for single precision
    real(kind=DP), public, parameter :: DP_NAN = &
                                        transfer(-2251799813685248_int64, 1._DP)
    !! NaN for double precision

    ! define GENE-X precision
#ifdef DOUBLE_PREC
    integer, public, parameter :: GP = DP
    !! GENE-X Fortran precision
    integer, public, parameter :: CP = C_DOUBLE
    !! GENE-X C/C++ precision
    integer, public, parameter :: MPI_GP = MPI_DOUBLE
    !! GENE-X precision for MPI
    integer, public, parameter :: NF90_GP = NF90_DOUBLE
    !! GENE-X precision for NetCDF
    real(kind=GP), public, parameter :: GP_EPS = DP_EPS
    !! Machine epsilon for GENE-X precision
    real(kind=GP), public, parameter :: GP_NAN = DP_NAN
    !! NaN for GENE-X precision
#else
    integer, public, parameter :: GP = SP
    !! GENE-X Fortran precision
    integer, public, parameter :: CP = C_FLOAT
    !! GENE-X C/C++ precision
    integer, public, parameter :: MPI_GP = MPI_FLOAT
    !! GENE-X precision for MPI
    integer, public, parameter :: NF90_GP = NF90_FLOAT
    !! GENE-X precision for NetCDF
    real(kind=GP), public, parameter :: GP_EPS = SP_EPS
    !! Machine epsilon for GENE-X precision
    real(kind=GP), public, parameter :: GP_NAN = SP_NAN
    !! NaN for GENE-X precision
#endif

end module genex_fortran_env_m
