#include "device_memory_tracer.hxx"
#include "logger.hxx"
#include <cstdint>
#include <cassert>
#include <fstream>
#include <vector>

namespace device_memory_tracer
{
    // Private object with file scope and external linkage

    // ID number of the current active memory tracing region
    static uint32_t active_region_id = 0;

    // Region name of the current active memory tracing region
    static std::string active_region_name {};

    // Tracker for the current total memory usage
    static int64_t total_memory_usage = 0;

    // Array of trackers for each memory size of
    // allocation or deallocation in the same region
    static std::vector<int64_t> memory_size {};

    // Array of trackers for each memory usage in the same region
    static std::vector<int64_t> memory_usage {};

    // Initial timestamp used as reference for the time recording
    static double initial_timestamp;

    // Array of timestamps for each tracer call in the same region
    static std::vector<double> timestamps {};

    // Write-only logger for the trace reports on the device memory usage
    static logger_t tracefile;

    void initialize(const MPI_Comm comm, const std::string trace_dir,
                    const double init_tstamp)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        const std::string fname = trace_dir + std::string("report_")
                                + std::to_string(rank) + std::string(".csv");

        tracefile.open_file(fname);
        tracefile.write("Region ID", "Region name", "Memory size [B]",
                        "Total memory usage [B]", "Timestamp [s]");
        initial_timestamp = init_tstamp;
    }

    void finalize()
    {
        tracefile.close_file();
    }

    void start_region(const std::string& region_name)
    {
        assert(active_region_name.empty() &&
            "The previous region is still active!");
        assert(memory_size.empty()  && "Stored data should be empty!");
        assert(memory_usage.empty() && "Stored data should be empty!");
        assert(timestamps.empty()   && "Stored data should be empty!");

        active_region_name = region_name;
        active_region_id += 1;
    }

    void end_region(const std::string& region_name)
    {
        assert(region_name == active_region_name &&
            "Attempting to close different tracing region!");

        // Write traces within the region
        for (auto i = 0; i < memory_size.size(); i++)
        {
            tracefile.write(active_region_id, active_region_name,
                            memory_size[i], memory_usage[i], timestamps[i]);
        }

        // Clear entries within the region
        active_region_name.clear();
        memory_size.clear();
        memory_usage.clear();
        timestamps.clear();
    }

    void trace(const size_t type_size, const int64_t signed_length)
    {
        const int64_t mem_size = type_size * signed_length;

        total_memory_usage += mem_size;
        memory_size.push_back(mem_size);
        memory_usage.push_back(total_memory_usage);
        timestamps.push_back(MPI_Wtime() - initial_timestamp);
    }
}

void cbind_device_memory_tracer_init(const int32_t comm, const char* trace_dir,
                                     const double init_tstamp)
{
    device_memory_tracer::initialize(
        MPI_Comm_f2c(comm), std::string{trace_dir}, init_tstamp);
}

void cbind_device_memory_tracer_fin()
{
    device_memory_tracer::finalize();
}
