#ifndef PARAMS_GPU_OFFLOAD_HXX
#define PARAMS_GPU_OFFLOAD_HXX

#include "genex_cxx_env.hxx"

#ifdef __cplusplus
extern "C" {
#endif

struct params_gpu_offload_data_t
{
    int32_t use_gpu_offload;
    int32_t gpu_offload_backend;
    int32_t use_parallax_gpu_offload;
    int32_t parallax_gpu_offload_backend;
    int32_t use_parallax_gpu_data_explicit;
    int32_t swap_mesh_members;
    int32_t large_array_alignment;
};

#ifdef __cplusplus
}
#endif

// Namespace for parameters related to GPU offloading in GENE-X
namespace params_gpu_offload
{
    // Enumerator for GPU offload backend switch
    // NOTE: Synchronize with src/parameters/params_gpu_offload_m.f90
    enum class backend_t
    {
        CPU  = 0,
        ACC  = 1,
        OMPX = 2,
        CUDA = 3
    };

    // Default alignment for new expression by C++ standard
    constexpr std::align_val_t default_alignment =
        std::align_val_t { __STDCPP_DEFAULT_NEW_ALIGNMENT__ };

    // Getter that returns true if GPU offloading is used
    bool get_use_offload();

    // Getter for the chosen GPU offload backend
    backend_t get_backend();

    // Getter that returns true if mesh member pointer swap is specified
    bool get_swap_mesh_members();

    // Getter that returns true if custom alignment is specified
    bool get_use_array_alignment();

    // Getter for the custom alignment setting for large arrays
    std::align_val_t get_array_alignment();
}

// Namespace for parameters related to GPU offloading in PARALLAX
namespace params_parallax_gpu_offload
{
    // Enumerator for GPU offload and data backend switch for PARALLAX
    // NOTE: Synchronize with parallax/src/device_handling_m.f90
    enum class backend_t
    {
        CPU     = 0, // Associated with BACKEND_CPU
        GPU     = 1, // Associated with BACKEND_GPU
        RLU_CPU = 2, // Associated with BACKEND_ROCALUTION_CPU
        RLU_GPU = 3  // Associated with BACKEND_ROCALUTION_GPU
    };

    // Getter that returns true if GPU offloading for PARALLAX is used
    bool get_use_offload();

    // Getter for the chosen GPU offloading backend for PARALLAX
    backend_t get_backend();

    // Getter that returns true if GPU data for PARALLAX is used
    bool get_use_data_explicit();
}

#ifdef ENABLE_OPENACC
// Get number of GPUs found by OpenACC
int32_t get_num_devices_acc();

// Check if a kernel can be executed on GPUs by OpenACC
bool is_on_device_acc();
#endif

#ifdef ENABLE_OPENMPX
// Get number of GPUs found by OpenMP offload
int32_t get_num_devices_ompx();

// Check if a kernel can be executed on GPUs by OpenMP offload
bool is_on_device_ompx();
#endif

#ifdef ENABLE_CUDA
// Get number of GPUs found by CUDA
int32_t get_num_devices_cuda();

// Check if a kernel can be executed on GPUs by CUDA
bool is_on_device_cuda();
#endif

#ifdef __cplusplus
extern "C" {
#endif

void cbind_set_params_gpu_offload(
    struct params_gpu_offload_data_t* params_data);

int32_t cbind_check_gpu_functionalities();

#ifdef __cplusplus
}
#endif

#endif
