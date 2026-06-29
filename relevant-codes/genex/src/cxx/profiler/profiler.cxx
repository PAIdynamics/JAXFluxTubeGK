#include "profiler.hxx"
#include "params_devtools.hxx"
#include <unordered_map>
#include <mpi.h>

#ifdef ENABLE_NVTX
#include "annotation_nvtx.hxx"
#elif ENABLE_ROCTX
#include "annotation_roctx.hxx"
#elif ENABLE_DLB
#include "annotation_dlb.hxx"
#elif ENABLE_CALIPER
#include "annotation_caliper.hxx"
#else
#include "annotation.hxx"
#endif

namespace profiler
{
    // Private objects with file scope and external linkage

    // Hash map to the start timestamps given by each region names
    static std::unordered_map<std::string, double> start_timestamps;

    // Hash map to the time intervals given by each region names
    // NOTE: Multiple calls with the same region name append the list
    static std::unordered_map<std::string, double> time_intervals;

    // Hash map to the time intervals of the initial calls
    // given by each region names
    static std::unordered_map<std::string, double> initial_call_times;

    // Hash map to the number of consecutive calls given by each region names
    static std::unordered_map<std::string, int32_t> number_of_calls;

    // Routines for the private objects

    void start_region(const std::string& region_name)
    {
        is_erroneous = false;

        // Check for illegal characters in region name, i.e. ".", "/"
        if (region_name.find(".") != std::string::npos)
        {
            std::cerr << "C++ error: profiler region name " << region_name
                      << " contains illegal character '.'!"
                      << std::endl;
            is_erroneous = true;
        }
        else if (region_name.find("/") != std::string::npos)
        {
            std::cerr << "C++ error: profiler region name " << region_name
                      << " contains illegal character '/'!"
                      << std::endl;
            is_erroneous = true;
        }

        // Check if the profiler region is activated
        if (start_timestamps.find(region_name) != start_timestamps.end())
        {
            std::cerr << "C++ error: profiler region " << region_name
                      << " is already activated!"
                      << std::endl;
            is_erroneous = true;
        }
        else
        {
            annotation::start_region(region_name);
            double t_start = MPI_Wtime();
            start_timestamps[region_name] = t_start;
        }
    }

    void end_region(const std::string& region_name)
    {
        is_erroneous = false;

        // Check if the profiler region is activated
        if (start_timestamps.find(region_name) != start_timestamps.end())
        {
            // Store the time interval of the profiler region if the region is
            // already activated
            double t_end = MPI_Wtime();
            annotation::end_region(region_name);
            double t_interval = t_end - start_timestamps[region_name];
            time_intervals[region_name] += t_interval;

            // Check if this is the first call of this profiler region
            if (number_of_calls.find(region_name) == number_of_calls.end())
            {
                initial_call_times[region_name] = t_interval;
            }
            number_of_calls[region_name] += 1;
            start_timestamps.erase(region_name);
        }
        else
        {
            std::cerr << "C++ error: profiler region " << region_name
                      << " is not activated!"
                      << std::endl;
            is_erroneous = true;
        }
    }

    void start_allreduce_region(MPI_Comm comm)
    {
        // Isolate the synchronization time from the communication time
        // of MPI Allreduce
        if (params_devtools::get_isolate_tsync())
        {
            start_region("mpi_barrier");
            is_erroneous = is_erroneous || MPI_Barrier(comm);
            end_region("mpi_barrier");
        }

        start_region("mpi_allreduce");
    }

    void end_allreduce_region()
    {
        end_region("mpi_allreduce");
    }
}

void cbind_annotation_start(const char* region_name_c)
{
    annotation::start_region(std::string{region_name_c});
}

void cbind_annotation_end(const char* region_name_c)
{
    annotation::end_region(std::string{region_name_c});
}

int32_t cbind_get_profiler_region_time(const char* region_name_c, double* times,
                                       int32_t* n_calls)
{
    const std::string& region_name{region_name_c};
    // Check if the time interval of the profiler region is measured
    if (profiler::time_intervals.find(region_name)
        != profiler::time_intervals.end())
    {
        n_calls[0] = profiler::number_of_calls[region_name];
        times[0]   = profiler::time_intervals[region_name];
        // NOTE: initial_call_times is always passed to Fortran but is not
        //       always saved as the initial call time. Only on Fortran, it is
        //       checked if it is actually the initial call time.
        times[1]   = profiler::initial_call_times[region_name];

        // NOTE: All entries associated by region_name are erased by consecutive
        //       calls of C++ profiler::start_region(), profiler::end_region()
        //       and Fortran profiler_inject(). Overlapping calls are possible.
        //       This prevents the need to reset the C++ entries by Fortran
        //       profiler_reset() call. This also avoid mixups between two
        //       regions that have the same name but are on different branches
        //       of the profiler tree.
        profiler::number_of_calls.erase(region_name);
        profiler::time_intervals.erase(region_name);
        profiler::initial_call_times.erase(region_name);
    }
    else
    {
        std::cerr << "C++ error: the time measurements of profiler region "
                  << region_name << " cannot be found!"
                  << std::endl;
        profiler::is_erroneous = true;
    }
    return (int32_t) profiler::is_erroneous;
}
