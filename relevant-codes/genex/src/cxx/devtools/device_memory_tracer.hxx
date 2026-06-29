#ifndef DEVICE_MEMORY_TRACER_HXX
#define DEVICE_MEMORY_TRACER_HXX

#include <string>
#include <mpi.h>

// Namespace for the device memory usage tracer
namespace device_memory_tracer
{
    // Start a memory tracing region for the device memory usage
    void start_region(const std::string& region_name);

    // End a tracing region for device memory usage
    void end_region(const std::string& region_name);

    // Initialize the file stream for device memory usage trace reports
    void initialize(const MPI_Comm comm, const std::string trace_dir,
                    const double initial_timestamp);

    // Finalize the file stream for device memory usage trace reports
    void finalize();

    // Trace the memory allocated on and freed from the device
    void trace(const size_t type_size, const int64_t signed_length);
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_device_memory_tracer_init(const int32_t comm, const char* trace_dir,
                                     const double init_tstamp);

void cbind_device_memory_tracer_fin();

#ifdef __cplusplus
}
#endif

#endif
