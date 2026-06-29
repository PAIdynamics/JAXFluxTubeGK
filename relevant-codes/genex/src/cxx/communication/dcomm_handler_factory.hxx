#ifndef DCOMM_HANDLER_FACTORY_HXX
#define DCOMM_HANDLER_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "dcomm_handler.hxx"
#include "dcomm_handler_omp.hxx"

#ifdef ENABLE_OPENACC
#include "dcomm_handler_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "dcomm_handler_ompx.hxx"
#endif

namespace dcomm_handler
{
    // Factory method to instantiate dcomm_handler_t derived class
    // based on the chosen GPU offload backend
    dcomm_handler_t* create(const dcomm_handler_data_t* dcomm_handler_data)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new dcomm_handler_omp_t(dcomm_handler_data);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new dcomm_handler_acc_t(dcomm_handler_data);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new dcomm_handler_ompx_t(dcomm_handler_data);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
