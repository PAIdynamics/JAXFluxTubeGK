#ifndef DEVICE_MEMORY_DEBUGGER_HXX
#define DEVICE_MEMORY_DEBUGGER_HXX

#include "params_gpu_offload.hxx"
#include "params_devtools.hxx"
#include "device_memory_tracer.hxx"
#include <string>
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Namespace for the device memory usage debugger
// after allocation and deallocation
namespace device_memory_debugger
{
    // Enumerator for device memory usage mode
    enum class mode_t
    {
        ALLOC,
        DEALLOC
    };

    // Get the device memory usage mode
    mode_t get_mode();

    // Routine to check if USM is set
    bool check_usm();

    // Start a memory tracing region for the device memory usage
    void start_region(const std::string& region_name, const mode_t mode);

    // End a tracing region for device memory usage
    void end_region(const std::string& region_name);

    // Check if the device memory is set and used as intended. Return true if
    // the memory allocation or deallocation on device memory is erroneous.
    template <typename T>
    inline bool is_invalid(T* ptr, const uint64_t length)
    {
        if (length == 0)
        {
            return false;
        }

        // Get the memory addresses of the first and the last elemets
        // of the associated array
        void* head_dptr;
        void* tail_dptr;
        int32_t gpu_rank;
        switch (params_gpu_offload::get_backend())
        {
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                head_dptr = acc_deviceptr(ptr);
                tail_dptr = acc_deviceptr(ptr + length - 1);
                break;
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                gpu_rank  = omp_get_default_device();
                head_dptr = omp_get_mapped_ptr(ptr, gpu_rank);
                tail_dptr = omp_get_mapped_ptr(ptr + length - 1, gpu_rank);
                break;
#endif
            default:
                head_dptr = (void*) ptr;
                tail_dptr = (void*) (ptr + length - 1);
                break;
        }

        // Overriding the pointers when using Unified Shared Memory (USM)
        if (check_usm())
        {
            head_dptr = (void*) ptr;
            tail_dptr = (void*) (ptr + length - 1);
        }

        // Check the memory addresses
        bool is_present = (head_dptr != nullptr && tail_dptr != nullptr);

        // Check if the device memory tracing is enabled
        bool trace_dmem = params_devtools::get_trace_device_memory();

        // Status assertion and memory recording if it works as intended
        switch (get_mode())
        {
            case mode_t::ALLOC:
                if (trace_dmem && is_present)
                {
                    device_memory_tracer::trace(sizeof(T), (int64_t) length);
                }
                return !is_present;
            case mode_t::DEALLOC:
                if (trace_dmem && !is_present)
                {
                    device_memory_tracer::trace(sizeof(T),
                                                -1 * (int64_t) length);
                }
                return is_present;
            default:
                return true;
        }
    }

    // Support for void pointer is deleted because pointer arithmetic with void
    // is illegal
    template<>
    inline bool is_invalid<void>(void *ptr, const uint64_t length) = delete;
}

#endif
