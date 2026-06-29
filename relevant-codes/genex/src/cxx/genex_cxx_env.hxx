#ifndef GENEX_CXX_ENV_HXX
#define GENEX_CXX_ENV_HXX

#include <cstdlib>
//#include <cstdint>
#include <cinttypes>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include <iostream>
#include <mpi.h>
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DOUBLE_PREC

// GENE-X C/C++ double precision corresponds to real(kind=GP) in Fortran
typedef double real_t;
// Machine epsilon for GENE-X precision
const double real_eps = std::numeric_limits<double>::epsilon();
// GENE-X C/C++ double precision for MPI
const MPI_Datatype MPI_REAL_T = MPI_DOUBLE;

#else

// GENEX C/C++ single precision corresponds to real(kind=GP) in Fortran
typedef float real_t;
// Machine epsilon for GENE-X precision
const float real_eps = std::numeric_limits<float>::epsilon();
// GENE-X C/C++ single precision for MPI
const MPI_Datatype MPI_REAL_T = MPI_FLOAT;

#endif

#ifdef __cplusplus
}
#endif

#endif
