#ifndef OP_SET_UNIFORM_FACTORY_HXX
#define OP_SET_UNIFORM_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_set_uniform.hxx"
#include "op_set_uniform_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_set_uniform_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_set_uniform_ompx.hxx"
#endif

#ifdef ENABLE_CUDA
#include "op_set_uniform_cuda.cuh"
#endif

namespace op_set_uniform_gpu
{
    // Factory method to instantiate op_set_uniform_gpu_t derived class
    // based on the chosen GPU offload backend
    inline op_set_uniform_gpu_t* create()
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_set_uniform_omp_t();
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_set_uniform_acc_t();
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_set_uniform_ompx_t();
#endif
#ifdef ENABLE_CUDA
            case params_gpu_offload::backend_t::CUDA:
                return new op_set_uniform_cuda_t();
#endif
            default:
                return nullptr;
        }
    }
}

#endif
