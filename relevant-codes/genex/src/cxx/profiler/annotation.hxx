#ifndef ANNOTATION_HXX
#define ANNOTATION_HXX

#include <iostream>

namespace annotation
{
    // Wrapper functions to annotate the start of a profiler region
    // NOTE: The base version without special annotation does nothing
    inline void start_region(const std::string& region_name) {}

    // Wrapper functions to annotate the end of a profiler region
    // NOTE: The base version without special annotation does nothing
    inline void end_region(const std::string& region_name) {}
}
#endif
