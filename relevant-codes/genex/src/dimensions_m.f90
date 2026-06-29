module dimensions_m
    !! Module defining the order of the dimensions for the 5D and 2D datatypes
    public :: DIM_RZ, DIM_PHI, DIM_VP, DIM_MU, DIM_SP

    enum, bind(C)
        !! Enum defining the order of the dimensions for 5D and 2D datatypes
        !! The enum should always start at 1 and be continuous such that we can
        !! loop over indices
        !! NOTE: Synchronize with src/cxx/dimensions.hxx file
        enumerator :: DIM_RZ  = 1
        enumerator :: DIM_PHI = 2
        enumerator :: DIM_VP  = 3
        enumerator :: DIM_MU  = 4
        enumerator :: DIM_SP  = 5
    end enum
end module
