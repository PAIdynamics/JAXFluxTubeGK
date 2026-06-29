#ifndef HELPERS_PROFILER_CXX
#define HELPERS_PROFILER_CXX

#include "profiler.hxx"
#include <time.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_profiler_injection(const int32_t t_sleep, const char* region_name,
                                 const int32_t n_calls)
{
    int32_t ierr = 1;
    struct timespec request, remaining;

    // Convert the requested sleep time t_sleep [ms] to request.tv_sec [s]
    // and request.tv_nsec [ns]
    request.tv_sec  = t_sleep / 1000;
    request.tv_nsec = (t_sleep % 1000) * 1000000L;

    profiler::start_region(std::string{region_name});

    // Suspend the execution time by the requested sleep time
    int32_t response = nanosleep(&request , &remaining);
    if (response == 0)
    {
        ierr = 0; // Successful sleep
    }

    profiler::end_region(std::string{region_name});

    // Generate multiple calls of the same profiler region and each calls last
    // 10 ms
    request.tv_sec  = 0;
    request.tv_nsec = 10000000L;
    for (int32_t i = 0; i < n_calls; i++)
    {
        profiler::start_region("multi_call");

        ierr = 1;
        struct timespec remaining_multi;
        int32_t response_multi = nanosleep(&request, &remaining_multi);
        if (response_multi == 0)
        {
            ierr = 0; // Successful sleep
        }

        profiler::end_region("multi_call");
    }

    return (int32_t) ierr;
}

// Create a dummy mpi_allreduce profiling region
void cbind_profiler_allreduce(const int32_t comm)
{
    profiler::start_allreduce_region(MPI_Comm_f2c(comm));
    profiler::end_allreduce_region();
}

#ifdef __cplusplus
}
#endif

#endif
