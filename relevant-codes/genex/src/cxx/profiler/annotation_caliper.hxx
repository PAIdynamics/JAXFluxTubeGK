#ifndef ANNOTATION_CALIPER_HXX
#define ANNOTATION_CALIPER_HXX

#include <iostream>
#include <caliper/cali.h>

namespace annotation
{
    // Wrapper functions to annotate the start of a profiler region
    // NOTE: This uses Caliper library
    inline void start_region(const std::string& region_name)
    {
        CALI_MARK_BEGIN(region_name.c_str());
    }

    // Wrapper functions to annotate the end of a profiler region
    // NOTE: This uses Caliper library.
    inline void end_region(const std::string& region_name)
    {
        CALI_MARK_END(region_name.c_str());
    }
}
#endif
