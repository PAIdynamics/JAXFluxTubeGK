#ifndef MAILBOX_FACTORY_HXX
#define MAILBOX_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "mailbox.hxx"
#include "mailbox_omp.hxx"

#ifdef ENABLE_OPENACC
#include "mailbox_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "mailbox_ompx.hxx"
#endif

namespace mailbox
{
    // Factory method to instantiate mailbox_t derived class
    // based on the chosen GPU offload backend
    inline mailbox_t* create(const int32_t n_mailbox_cells,
                             const int32_t max_n_mail_partners)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new mailbox_omp_t(n_mailbox_cells, max_n_mail_partners);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new mailbox_acc_t(n_mailbox_cells, max_n_mail_partners);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new mailbox_ompx_t(n_mailbox_cells, max_n_mail_partners);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
