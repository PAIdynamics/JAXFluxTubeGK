module genex_status_codes_m
    !! Defines status codes that are used to denote errors and warnings

    implicit none
    public

    enum, bind(C)
        enumerator :: GENEX_SUCCESS             =   0

        enumerator :: GENEX_ERR_ARGUMENTS       =  -1
        !! Arguments provided to the program are incorrect
        enumerator :: GENEX_ERR_NETCDF          =  -2
        !! Executing netCDF command failed
        enumerator :: GENEX_ERR_MAIN            =  -3
        !! Error in GENEX main program
        enumerator :: GENEX_ERR_PARAMETER_FILE  =  -4
        !! I/O error in one of the parameter files
        enumerator :: GENEX_ERR_DATA_STRUCTURE  =  -5
        !! Error in datastructures
        enumerator :: GENEX_ERR_PARALLAX        =  -6
        !! Error in the PARALLAX library
        enumerator :: GENEX_ERR_COMMUNICATION   =  -7
        !! Error in communication or MPI domain decomposition
        enumerator :: GENEX_ERR_MAIL_DELIVERY   =  -8
        !! Error in delivering mail in MPI exchange
        enumerator :: GENEX_ERR_STATE_VECTOR    =  -9
        !! Error in the state vector
        enumerator :: GENEX_ERR_BENCHMARKS      = -10
        !! Error in the benchmark program
        enumerator :: GENEX_ERR_DIAGNOSTICS     = -11
        !! Error in calculating diagnostics
        enumerator :: GENEX_ERR_TIMESTEP        = -12
        !! Error in the timestep
        enumerator :: GENEX_ERR_PROFILER        = -13
        !! Error in the profiler
        enumerator :: GENEX_ERR_MESH            = -14
        !! Error in the mesh
        enumerator :: GENEX_ERR_UTESTS          = -15
        !! Error in a unit test
        enumerator :: GENEX_ERR_COLL            = -16
        !! Error in a collision operator
        enumerator :: GENEX_ERR_FIELD_SOLVE     = -17
        !! Error in the field solve
        enumerator :: GENEX_ERR_DIST_INITIAL    = -18
        !! Error in the initial distribution function
        enumerator :: GENEX_ERR_MMS             = -19
        !! Error in the MMS test
        enumerator :: GENEX_ERR_SYSTEM_CALL     = -20
        !! Error in executing a system call
        enumerator :: GENEX_ERR_MATH            = -21
        !! Error in a mathematical algorithm
        enumerator :: GENEX_ERR_GPU             = -22
        !! GPU specific error
        enumerator :: GENEX_ERR_DIAG_FILES      = -23
        !! Error in diagnostics or checkpoint files
        enumerator :: GENEX_ERR_OPERATORS       = -24
        !! Error in an operator
        enumerator :: GENEX_ERR_PARAMETERS      = -25
        !! Invalid (combination of) parameter(s)
        enumerator :: GENEX_ERR_BSG             = -26
        !! Error in BSG plugin
        enumerator :: GENEX_ERR_NOT_IMPLEMENTED = -27
        !! Thrown if a not implemented feature was executed on runtime
        enumerator :: GENEX_ERR_VSPEC           = -28
        !! Vspec specific error

        enumerator :: GENEX_WRN_GENERAL         =   1
        !! General warning. Should not be used if a more specific code exists.
        enumerator :: GENEX_WRN_FILE_NOT_EXIST  =   2
        !! Warning to indicate that a file does not exist
        enumerator :: GENEX_WRN_RESIDUUM_NAN    =   3
        !! Warning that a solver residuum was NaN
        enumerator :: GENEX_WRN_FIELD_SOLVE     =   4
        !! Warning in the field solver
        enumerator :: GENEX_WRN_TIMESTEP        =   5
        !! Warning in the time step
        enumerator :: GENEX_WRN_PROFILER        =   6
        !! Warning in the profiler
        enumerator :: GENEX_WRN_COLL            =   7
        !! Warning in a collision operator
        enumerator :: GENEX_WRN_MESH            =   8
        !! Warning in the mesh
        enumerator :: GENEX_WRN_SYSTEM_CALL     =   9
        !! Warning when executing a system call
        enumerator :: GENEX_WRN_DATA_STRUCTURE  =  10
        !! Warning in datastructures
        enumerator :: GENEX_WRN_PARAMETERS      =  11
        !! Warning of unusually set parameters
    end enum
end module
