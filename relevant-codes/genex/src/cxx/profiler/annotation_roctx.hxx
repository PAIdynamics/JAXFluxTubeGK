#ifndef ANNOTATION_ROCTX_HXX
#define ANNOTATION_ROCTX_HXX

#include <iostream>
#include <rocprofiler-sdk-roctx/roctx.h>

namespace annotation
{
    // Wrapper functions to annotate the start of a profiler region
    // NOTE: This uses ROC-TX Library.
    inline void start_region(const std::string& region_name)
    {
        roctxRangePush(region_name.c_str());
    }

    // Wrapper functions to annotate the end of a profiler region
    // NOTE: This uses ROC-TX Library.
    //       Region name is not needed but it increases readability to users.
    inline void end_region(const std::string& region_name)
    {
        roctxRangePop();
    }
}
#endif
