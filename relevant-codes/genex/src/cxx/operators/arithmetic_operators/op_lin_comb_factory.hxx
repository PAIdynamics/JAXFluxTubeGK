#ifndef OP_LIN_COMB_FACTORY_HXX
#define OP_LIN_COMB_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_lin_comb.hxx"
#include "op_lin_comb_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_lin_comb_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_lin_comb_ompx.hxx"
#endif

#ifdef ENABLE_CUDA
#include "op_lin_comb_cuda.cuh"
#endif

namespace op_lin_comb_gpu
{
    // Factory method to instantiate op_lin_comb_gpu_t derived class
    // based on the chosen GPU offload backend
    op_lin_comb_gpu_t* create()
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_lin_comb_omp_t();
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_lin_comb_acc_t();
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_lin_comb_ompx_t();
#endif
#ifdef ENABLE_CUDA
            case params_gpu_offload::backend_t::CUDA:
                return new op_lin_comb_cuda_t();
#endif
            default:
                return nullptr;
        }
    }
}

#endif
