#ifndef DIMENSIONS_HXX
#define DIMENSIONS_HXX

// Enum defining the order of the dimensions for 5D and 2D datatypes
// The enum should always start at 1 and be continuous such that we can
// loop over indices
// NOTE: Synchronize with src/dimensions_m.f90 file
enum dimensions
{
    DIM_RZ  = 1,
    DIM_PHI = 2,
    DIM_VP  = 3,
    DIM_MU  = 4,
    DIM_SP  = 5
};

#endif
