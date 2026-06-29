#ifndef OP_COPY_FACTORY_HXX
#define OP_COPY_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_copy.hxx"
#include "op_copy_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_copy_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_copy_ompx.hxx"
#endif

#ifdef ENABLE_CUDA
#include "op_copy_cuda.cuh"
#endif

namespace op_copy_gpu
{
    // Factory method to instantiate op_copy_gpu_t derived class
    // based on the chosen GPU offload backend
    op_copy_gpu_t* create()
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_copy_omp_t();
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_copy_acc_t();
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_copy_ompx_t();
#endif
#ifdef ENABLE_CUDA
            case params_gpu_offload::backend_t::CUDA:
                return new op_copy_cuda_t();
#endif
            default:
                return nullptr;
        }
    }
}

#endif
