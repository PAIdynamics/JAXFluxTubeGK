#ifndef ANNOTATION_NVTX_HXX
#define ANNOTATION_NVTX_HXX

#include <iostream>
#include <nvtx3/nvToolsExt.h>

namespace annotation
{
    // Wrapper functions to annotate the start of a profiler region
    // NOTE: This uses NVTX Tools Extension Library
    // TODO: Customize NVTX ranges with color, etc
    inline void start_region(const std::string& region_name)
    {
        nvtxRangePushA(region_name.c_str());
    }

    // Wrapper functions to annotate the end of a profiler region
    // NOTE: This uses NVTX Tools Extension Library.
    //       Region name is not needed but it increases readability to users.
    inline void end_region(const std::string& region_name)
    {
        nvtxRangePop();
    }
}
#endif
