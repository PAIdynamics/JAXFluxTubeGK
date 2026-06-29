#ifndef LOGGER_HXX
#define LOGGER_HXX

#include <fstream>
#include <string>
#include <optional>
#include <cassert>
#include <iomanip>
#include <limits>
#include <type_traits>

// Log handler for outputs to a file
class logger_t
{
private:
    // Write-only file stream
    std::ofstream file;
    // Full path and name of the logger file
    std::string filename {};
    // Separator
    std::string separator {","};
    // True if this MPI process is allowed to print
    bool allow_print = false;

    // Common helper function to write a single entry
    // with full precision for doubles
    template<typename T>
    inline void write_full_precision(T&& entry)
    {
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            // Save current precision, set to max_digits10 for doubles
            auto old_precision = file.precision(
                std::numeric_limits<std::decay_t<T>>::max_digits10);
            file << std::forward<T>(entry);
            // Restore original precision
            file.precision(old_precision);
        } else {
            // Normal output for non-doubles
            file << std::forward<T>(entry);
        }
    }

public:
    // Default constructor
    logger_t() = default;

    // Constructor
    logger_t(
        const std::string& filename,
        const std::optional<std::string> separator = std::nullopt,
        const std::optional<bool> print_only_master = std::nullopt,
        const std::optional<int32_t> mpi_rank = std::nullopt)
    {
        this->open_file(filename, separator, print_only_master, mpi_rank);
    }

    // Destructor
    ~logger_t()
    {
        this->close_file();
    }

    // Getter for the full path and name of the logger file
    const std::string& get_filename()
    {
        return this->filename;
    }

    // Open file helper
    void open_file(
        const std::string& filename,
        const std::optional<std::string> separator = std::nullopt,
        const std::optional<bool> print_only_master = std::nullopt,
        const std::optional<int32_t> mpi_rank = std::nullopt)
    {
        this->separator = separator.value_or(this->separator);
        assert((!print_only_master || mpi_rank) &&
            "Argument mpi_rank is missing!");

        // When print_only_master is set to true, only master process can print.
        // Otherwise all MPI process can print.
        if (!print_only_master.value_or(false) || mpi_rank.value_or(1) == 0)
        {
            this->allow_print = true;
            // Create and open file, overwrite file if exists
            this->file.open(filename);
            this->filename = filename;
        }
    }

    // Close file helper
    void close_file()
    {
        if (this->file.is_open() && this->allow_print)
        {
            this->file.close();
        }
        // Reset members to their default value
        this->separator   = ",";
        this->allow_print = false;
    }

    // Base function handles single argument to write to the file
    template<typename T>
    inline void write(T&& entry)
    {
        write_full_precision(std::forward<T>(entry));
        this->file << std::endl;
    }

    // Recursive function handles multiple arguments to write to the file
    template<typename T, typename... Args>
    inline void write(T&& first_entry, Args&&... entries)
    {
        write_full_precision(std::forward<T>(first_entry));
        this->file << this->separator;
        write(std::forward<Args>(entries)...); // Recurse
    }
};

#endif
