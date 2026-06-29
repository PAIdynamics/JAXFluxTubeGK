#ifndef PROFILER_HXX
#define PROFILER_HXX

#include <iostream>
#include <optional>
#include <mpi.h>

namespace profiler
{
    // Error flag for the profiler
    inline bool is_erroneous = false;

    // Routine to start a nested profiling region and store the start timestamp
    void start_region(const std::string& region_name);

    // Routine to end a nested profiling region and store the time interval
    void end_region(const std::string& region_name);

    // Routine to start a profiling region for MPI_Allreduce and
    // store the start timestamp
    void start_allreduce_region(MPI_Comm comm);

    // Routine to end a nested profiling region for MPI_Allreduce and
    // store the time interval
    void end_allreduce_region();
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_annotation_start(const char* region_name_c);

void cbind_annotation_end(const char* region_name_c);

int32_t cbind_get_profiler_region_time(const char* region_name_c, double* times,
                                       int32_t* n_calls);

#ifdef __cplusplus
}
#endif

#endif
