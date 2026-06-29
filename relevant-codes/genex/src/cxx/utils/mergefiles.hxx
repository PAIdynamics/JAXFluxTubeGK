#ifndef MERGEFILES_HXX
#define MERGEFILES_HXX

#include <string>
#include <list>
#include <optional>
#include <fstream>
#include <iostream>
#include <cassert>

// Namespace for utils related to general file and folder handling
namespace fileutils
{
    // Merge the content of files given by a list into a new file
    void mergefiles(const std::list<std::string>& filelist,
                    const std::string& destination_filepath,
                    const bool append = false)
    {
        // Open the destination file
        std::ofstream write_file;
        if (append)
        {
            write_file.open(destination_filepath,
                            std::ios_base::binary | std::ios_base::app);
        }
        else
        {
            write_file.open(destination_filepath, std::ios_base::binary);
        }

        // Read each file and transfer the content to the destination file
        for (auto file : filelist)
        {
            std::ifstream read_file(file, std::ios_base::binary);
            write_file << read_file.rdbuf();
            read_file.close();
        }

        write_file.close();
    }
}

#endif
