#ifndef PARAMS_DEVTOOLS_HXX
#define PARAMS_DEVTOOLS_HXX

#include <cinttypes>

// Namespace for parameters related to the developer tools
namespace params_devtools
{
    // Getter for a flag to isolate the synchronization time in the profiler
    bool get_isolate_tsync();

    // Getter that returns true if device memory tracing is specified
    bool get_trace_device_memory();
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_set_params_devtools(const int32_t isolate_tsync,
                               const int32_t trace_dmem);

#ifdef __cplusplus
}
#endif

#endif
