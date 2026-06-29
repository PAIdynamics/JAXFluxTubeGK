#ifndef LISTDIR_HXX
#define LISTDIR_HXX

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
    // Return list containing the content names in the directory given by a path
    std::list<std::string> listdir(const std::string& path)
    {
        std::list<std::string> contents;

        for (const auto& entry : filesystem::directory_iterator(path))
        {
            contents.push_back(entry.path());
        }

        return contents;
    }
}

#endif
