#include "params_devtools.hxx"

namespace params_devtools
{
    // Private object with file scope and external linkage

    // True if the synchronization time in the profiler is isolated
    static bool isolate_tsync {};

    // True if the device memory tracing is specified
    static bool trace_device_memory = false;

    // Getter for the private object

    bool get_isolate_tsync() { return isolate_tsync; }
    bool get_trace_device_memory() { return trace_device_memory; }
}

void cbind_set_params_devtools(const int32_t isolate_tsync,
                               const int32_t trace_dmem)
{
    params_devtools::isolate_tsync = static_cast<bool>(isolate_tsync);
    params_devtools::trace_device_memory = static_cast<bool>(trace_dmem);
}
