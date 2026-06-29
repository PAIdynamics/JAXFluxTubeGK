#ifndef RM_HXX
#define RM_HXX

#include <string>
#include <list>

#if __has_include(<filesystem>)
#include <filesystem>
namespace filesystem = std::filesystem;

#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;

#endif

// Namespace for utils related to general file and folder handling
namespace fileutils
{
    // Remove a file or a directory
    void rm(const std::string& file)
    {
        filesystem::remove(file);
    }

    // Remove files or directories given by a list
    void rm(const std::list<std::string>& filelist)
    {
        for (auto file : filelist)
        {
            filesystem::remove(file);
        }
    }
}

#endif
