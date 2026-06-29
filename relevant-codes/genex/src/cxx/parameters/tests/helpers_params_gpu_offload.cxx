#ifndef HELPERS_PARAMS_GPU_OFFLOAD_CXX
#define HELPERS_PARAMS_GPU_OFFLOAD_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_gpu_offload(
    struct params_gpu_offload_data_t* params_data)
{
    // NOTE: Use std::to_underlying() to cast enum class if C++23 is used

    params_data->use_gpu_offload     = params_gpu_offload::get_use_offload();
    params_data->gpu_offload_backend =
        static_cast<int32_t>(params_gpu_offload::get_backend());
    params_data->swap_mesh_members   =
        params_gpu_offload::get_swap_mesh_members();
    if(params_gpu_offload::get_array_alignment() ==
       params_gpu_offload::default_alignment)
    {
        params_data->large_array_alignment = 0;
    }
    else
    {
        std::align_val_t align = params_gpu_offload::get_array_alignment();
        params_data->large_array_alignment = static_cast<int32_t>(align);
    }

    params_data->use_parallax_gpu_offload =
        params_parallax_gpu_offload::get_use_offload();
    params_data->parallax_gpu_offload_backend =
        static_cast<int32_t>(params_parallax_gpu_offload::get_backend());
    params_data->use_parallax_gpu_data_explicit =
        params_parallax_gpu_offload::get_use_data_explicit();
}

#ifdef __cplusplus
}
#endif

#endif
