#include "device_memory_debugger.hxx"
#include <cstdlib>

namespace device_memory_debugger
{
    // Private object with file scope and external linkage

    // Current device memory usage mode
    static mode_t dmem_mode;

    // True if the USM runtime status is checked
    static bool is_usm_checked = false;

    // True if USM is set
    static bool is_usm_set = false;

    mode_t get_mode()
    {
        return dmem_mode;
    }

    bool check_usm()
    {
        if (is_usm_checked) return is_usm_set;

        if (const char *hsa_xnack = getenv("HSA_XNACK"))
        {
            if (std::string(hsa_xnack) == "1")
            {
                is_usm_set = true;
            }
        }
        is_usm_checked = true;
        return is_usm_set;
    }

    void start_region(const std::string& region_name, mode_t mode)
    {
        device_memory_tracer::start_region(region_name);
        dmem_mode = mode;
    }

    void end_region(const std::string& region_name)
    {
        device_memory_tracer::end_region(region_name);
    }
}
