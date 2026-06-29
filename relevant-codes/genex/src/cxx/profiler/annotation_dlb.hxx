#ifndef ANNOTATION_HXX
#define ANNOTATION_HXX

#include <iostream>
#include "dlb_talp.h"

namespace annotation
{
    // Wrapper functions to annotate the start of a profiler region
    inline void start_region(const std::string& region_name) {
        auto handle = DLB_MonitoringRegionRegister(region_name.c_str());
        DLB_MonitoringRegionStart(handle);
    }

    // Wrapper functions to annotate the end of a profiler region
    inline void end_region(const std::string& region_name) {
        auto handle = DLB_MonitoringRegionRegister(region_name.c_str());
        DLB_MonitoringRegionStop(handle);
    }
}
#endif
