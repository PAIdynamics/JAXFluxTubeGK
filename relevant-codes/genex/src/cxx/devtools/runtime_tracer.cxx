#include "runtime_tracer.hxx"
#include "logger.hxx"
#include "listdir.hxx"
#include "mergefiles.hxx"
#include "rm.hxx"

#include <unordered_map>
#include <cassert>
#include <algorithm>

namespace runtime_tracer
{
    // Private object with file scope and external linkage

    // Initial timestamp used as reference for the time recording
    static double init_tstamp;

    // Write-only logger for the trace reports on the profiling regions
    static logger_t tracefile;

    // Output directory
    static std::string output_dir;

    // Folder name of the trace files
    static std::string trace_folder_name;

    // MPI process ID
    static int32_t rank;

    // Hash map to the start timestamps given by each region names
    static std::unordered_map<std::string, double> start_tstamps;

    void initialize(const MPI_Comm comm, const std::string out_dir,
                    const std::string trace_folder,
                    const double initial_timestamp)
    {
        output_dir = out_dir;
        trace_folder_name = trace_folder;
        MPI_Comm_rank(comm, &rank);

        const std::string fname = output_dir + trace_folder_name
                                + std::string("/report_")
                                + std::to_string(rank) + std::string(".csv");

        tracefile.open_file(fname);
        init_tstamp = initial_timestamp;
    }

    void finalize(const MPI_Comm comm)
    {
        assert(start_tstamps.empty() &&
               "Some profiler regions are not properly ended!");
        tracefile.close_file();

        int32_t ierr = MPI_Barrier(comm);
        // Read all trace files and merge them
        if (rank == 0)
        {
            // Write the header to the main trace file
            std::string mainfile_path = output_dir
                                      + std::string("traces_runtime.csv");
            logger_t mainfile(mainfile_path);
            mainfile.write("Rank", "Region name", "Region path", "Region level",
                "Start timestamp [s]", "End timestamp [s]", "Duration [s]");
            mainfile.close_file();

            // List the trace files and merge them to the main trace file
            std::list<std::string> tracefiles =
                fileutils::listdir(output_dir + trace_folder_name);
            fileutils::mergefiles(tracefiles, mainfile_path, true);

            // Remove temporary trace files and folder
            fileutils::rm(tracefiles);
            fileutils::rm(output_dir + trace_folder_name);
        }
        ierr = MPI_Barrier(comm);
    }

    void trace(const mode_t mode, const std::string& region_path,
               const std::string& region_name, const double tstamp)
    {
        // Minor adjustment for region_path
        std::string path {region_path};
        path.erase(0, 6); // Remove the top parent "genex/" path
        if ((path == "genex") && (region_name == "genex"))
        {
            path = "";
        }

        if (mode == mode_t::START)
        {
            // Save the start timestamp to the hash map
            start_tstamps[path + region_name] = tstamp;
        }
        else
        {
            assert((start_tstamps.find(path + region_name) !=
                    start_tstamps.end()) &&
                   "The profiler regions are not properly started!");

            double start_tstamp = start_tstamps[path + region_name];
            start_tstamps.erase(path + region_name);

            // Calculate region level
            // NOTE: Use std::ranges::count if C++20 is used
            int32_t region_level = 0;
            if (!path.empty())
            {
                std::string::difference_type lvl =
                    std::count(path.begin(), path.end(), '/');
                region_level = static_cast<int32_t>(lvl) + 1;
            }

            // Print the trace information of the profiling region to
            // the trace file
            // Columns: Rank, Region name, Region path, Region level, Start
            //          timestamp [s], End timestamp [s], Duration [s]
            tracefile.write(rank, region_name, path, region_level,
                            start_tstamp - init_tstamp, tstamp - init_tstamp,
                            tstamp - start_tstamp);
        }
    }
}

void cbind_runtime_tracer_init(const int32_t comm, const char* out_dir,
                               const char* trace_folder,
                               const double init_tstamp)
{
    runtime_tracer::initialize(MPI_Comm_f2c(comm), std::string{out_dir},
                               std::string{trace_folder}, init_tstamp);
}

void cbind_runtime_tracer_fin(const int32_t comm)
{
    runtime_tracer::finalize(MPI_Comm_f2c(comm));
}

void cbind_runtime_tracer_trace(const int32_t mode, const char* region_path,
                                const char* region_name, const double tstamp)
{
    runtime_tracer::trace(
        runtime_tracer::mode_t{mode}, std::string{region_path},
        std::string{region_name}, tstamp);
}
