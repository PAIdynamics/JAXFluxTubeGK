#include "params_gpu_offload.hxx"
#include <cstddef>

namespace params_gpu_offload
{
    // Private object with file scope and external linkage

    // True if the Fortran/C++ interoperability in GENE-X is used
    static bool use_offload = false;

    // GENE-X GPU offload backend setup
    static backend_t backend = backend_t::CPU;

    // True if the swapping of mesh member pointers is specified
    static bool swap_mesh_members = false;

    // True if the custom array alignment is specified
    static bool use_array_alignment = false;

    // Custom alignment setting for large arrays
    static std::align_val_t large_array_alignment = default_alignment;

    // Getters for the private objects

    bool get_use_offload() { return use_offload; }
    backend_t get_backend() { return backend; }
    bool get_swap_mesh_members() { return swap_mesh_members; }
    bool get_use_array_alignment() { return use_array_alignment; }
    std::align_val_t get_array_alignment() { return large_array_alignment; }
}

namespace params_parallax_gpu_offload
{
    // Private objects with file scope and external linkage

    // True if the Fortran/C++ interoperability in PARALLAX is used
    static bool use_offload = false;

    // PARALLAX general GPU offload backend setup
    static backend_t backend = backend_t::CPU;

    // True if the advanced explicit data management in PARALLAX is used
    static bool use_data_explicit = false;

    // Getters for the private objects

    bool get_use_offload() { return use_offload; }
    backend_t get_backend() { return backend; }
    bool get_use_data_explicit() { return use_data_explicit; }
}

void cbind_set_params_gpu_offload(struct params_gpu_offload_data_t* params_data)
{
    params_gpu_offload::use_offload = params_data->use_gpu_offload;
    int32_t backend = params_data->gpu_offload_backend;
    params_gpu_offload::backend = params_gpu_offload::backend_t{backend};
    params_gpu_offload::swap_mesh_members = params_data->swap_mesh_members;
    if(params_data->large_array_alignment != 0)
    {
        long unsigned int align = params_data->large_array_alignment;
        params_gpu_offload::large_array_alignment = std::align_val_t{align};
        params_gpu_offload::use_array_alignment = true;
    }

    params_parallax_gpu_offload::use_offload =
        params_data->use_parallax_gpu_offload;
    backend = params_data->parallax_gpu_offload_backend;
    params_parallax_gpu_offload::backend =
        params_parallax_gpu_offload::backend_t{backend};
    params_parallax_gpu_offload::use_data_explicit =
        params_data->use_parallax_gpu_data_explicit;
}

int32_t cbind_check_gpu_functionalities()
{
    // Check if GPU devices can be found and tested
    // Return 0 if success or otherwise return 1
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return 0;
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return (int32_t) !((get_num_devices_acc() >= 1) &&
                                is_on_device_acc());
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return (int32_t) !((get_num_devices_ompx() >= 1) &&
                                is_on_device_ompx());
#endif
#ifdef ENABLE_CUDA
        case params_gpu_offload::backend_t::CUDA:
            return (int32_t) !(get_num_devices_cuda() >= 1) &&
                               is_on_device_cuda();
#endif
        default:
            return 1;
    }
}
