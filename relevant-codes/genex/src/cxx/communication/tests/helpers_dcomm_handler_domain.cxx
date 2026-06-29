#ifndef HELPERS_DCOMM_HANDLER_DOMAIN_CXX
#define HELPERS_DCOMM_HANDLER_DOMAIN_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "dcomm_handler.hxx"
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Copy the class members related to the domain decomposition from the source
// dcomm_handler object to the target dcomm_handler struct via OpenMP on CPU
int32_t dcomm_handler_domain_copy_cxx(
    const dcomm_handler_t& dcomm_handler_src,
    struct dcomm_handler_data_t* dcomm_handler_data_tgt)
{
    // Assign array member pointer related to the domain decomposition
    // to the target dcomm_handler
    int32_t* dim_permut_tgt         = dcomm_handler_data_tgt->dim_permut_ptr;
    int32_t* num_data_elements_tgt  =
        dcomm_handler_data_tgt->number_of_data_elements_ptr;
    int32_t* num_elements_tgt = dcomm_handler_data_tgt->number_of_elements_ptr;
    int32_t* num_ghost_tgt    = dcomm_handler_data_tgt->number_of_ghosts_ptr;
    int32_t* lb_tgt           = dcomm_handler_data_tgt->lb_ptr;
    int32_t* ub_tgt           = dcomm_handler_data_tgt->ub_ptr;
    int32_t* lb_stripped_tgt  = dcomm_handler_data_tgt->lb_stripped_ptr;
    int32_t* ub_stripped_tgt  = dcomm_handler_data_tgt->ub_stripped_ptr;

    // Copy the class members related to the domain decomposition
    // from the source to the target op via OpenMP
    #pragma omp parallel default(none) \
                         shared(dcomm_handler_src, dim_permut_tgt) \
                         shared(num_data_elements_tgt, num_elements_tgt) \
                         shared(num_ghost_tgt, lb_tgt, ub_tgt) \
                         shared(lb_stripped_tgt, ub_stripped_tgt)
    {
        #pragma omp for simd schedule(static) nowait
        for (int32_t n = 1; n <= dcomm_handler_src.get_n_dims(); n++)
        {
            dim_permut_tgt[n-1]        = dcomm_handler_src.get_dim_permut(n);
            num_data_elements_tgt[n-1] =
                dcomm_handler_src.get_num_data_elements(n);
            num_elements_tgt[n-1] = dcomm_handler_src.get_num_elements(n);
            num_ghost_tgt[n-1]    = dcomm_handler_src.get_num_ghosts(n);
            lb_tgt[n-1]           = dcomm_handler_src.get_lbound(n);
            ub_tgt[n-1]           = dcomm_handler_src.get_ubound(n);
            lb_stripped_tgt[n-1]  = dcomm_handler_src.get_lbound_stripped(n);
            ub_stripped_tgt[n-1]  = dcomm_handler_src.get_ubound_stripped(n);
        }
    }

    return 0;
}

#ifdef ENABLE_OPENACC

// Copy the class members related to the domain decomposition from the source
// dcomm_handler object to the target dcomm_handler struct via OpenACC on GPU
int32_t dcomm_handler_domain_copy_acc(
    const dcomm_handler_t& dcomm_handler_src,
    struct dcomm_handler_data_t* dcomm_handler_data_tgt)
{
    int32_t ndims = dcomm_handler_src.get_n_dims();

    // Assign array member pointer related to the domain decomposition
    // to the target dcomm_handler
    int32_t* dim_permut_tgt         = dcomm_handler_data_tgt->dim_permut_ptr;
    int32_t* num_data_elements_tgt  =
        dcomm_handler_data_tgt->number_of_data_elements_ptr;
    int32_t* num_elements_tgt = dcomm_handler_data_tgt->number_of_elements_ptr;
    int32_t* num_ghost_tgt    = dcomm_handler_data_tgt->number_of_ghosts_ptr;
    int32_t* lb_tgt           = dcomm_handler_data_tgt->lb_ptr;
    int32_t* ub_tgt           = dcomm_handler_data_tgt->ub_ptr;
    int32_t* lb_stripped_tgt  = dcomm_handler_data_tgt->lb_stripped_ptr;
    int32_t* ub_stripped_tgt  = dcomm_handler_data_tgt->ub_stripped_ptr;

    // Allocate and copy data from the host to the device
    #pragma acc enter data create(dim_permut_tgt[:ndims], \
                                  num_data_elements_tgt[:ndims], \
                                  num_elements_tgt[:ndims], \
                                  num_ghost_tgt[:ndims], \
                                  lb_tgt[:ndims], \
                                  ub_tgt[:ndims], \
                                  lb_stripped_tgt[:ndims], \
                                  ub_stripped_tgt[:ndims])

    // Copy the class members related to the domain decomposition
    // from the source to the target op via OpenACC
    #pragma acc parallel default(none) \
            present(dcomm_handler_src, dim_permut_tgt) \
            present(num_data_elements_tgt, num_elements_tgt) \
            present(num_ghost_tgt, lb_tgt, ub_tgt) \
            present(lb_stripped_tgt, ub_stripped_tgt)
    {
        #pragma acc loop independent
        for (int32_t n = 1; n <= dcomm_handler_src.get_n_dims(); n++)
        {
            dim_permut_tgt[n-1]        = dcomm_handler_src.get_dim_permut(n);
            num_data_elements_tgt[n-1] =
                dcomm_handler_src.get_num_data_elements(n);
            num_elements_tgt[n-1] = dcomm_handler_src.get_num_elements(n);
            num_ghost_tgt[n-1]    = dcomm_handler_src.get_num_ghosts(n);
            lb_tgt[n-1]           = dcomm_handler_src.get_lbound(n);
            ub_tgt[n-1]           = dcomm_handler_src.get_ubound(n);
            lb_stripped_tgt[n-1]  = dcomm_handler_src.get_lbound_stripped(n);
            ub_stripped_tgt[n-1]  = dcomm_handler_src.get_ubound_stripped(n);
        }
    }

    // Copy data from the device to the host
    #pragma acc exit data copyout(dim_permut_tgt[:ndims], \
                                  num_data_elements_tgt[:ndims], \
                                  num_elements_tgt[:ndims], \
                                  num_ghost_tgt[:ndims], \
                                  lb_tgt[:ndims], \
                                  ub_tgt[:ndims], \
                                  lb_stripped_tgt[:ndims], \
                                  ub_stripped_tgt[:ndims])

    return 0;
}

#endif

#ifdef ENABLE_OPENMPX

// Copy the class members related to the domain decomposition from the source
// dcomm_handler object to the target dcomm_handler struct
// via OpenMP offload on GPU
int32_t dcomm_handler_domain_copy_ompx(
    const dcomm_handler_t& dcomm_handler_src,
    struct dcomm_handler_data_t* dcomm_handler_data_tgt)
{
    int32_t ndims = dcomm_handler_src.get_n_dims();

    // Assign array member pointer related to the domain decomposition
    // to the target dcomm_handler
    int32_t* dim_permut_tgt         = dcomm_handler_data_tgt->dim_permut_ptr;
    int32_t* num_data_elements_tgt  =
        dcomm_handler_data_tgt->number_of_data_elements_ptr;
    int32_t* num_elements_tgt = dcomm_handler_data_tgt->number_of_elements_ptr;
    int32_t* num_ghost_tgt    = dcomm_handler_data_tgt->number_of_ghosts_ptr;
    int32_t* lb_tgt           = dcomm_handler_data_tgt->lb_ptr;
    int32_t* ub_tgt           = dcomm_handler_data_tgt->ub_ptr;
    int32_t* lb_stripped_tgt  = dcomm_handler_data_tgt->lb_stripped_ptr;
    int32_t* ub_stripped_tgt  = dcomm_handler_data_tgt->ub_stripped_ptr;

    // Allocate and copy data from the host to the device
    #pragma omp target enter data map(to: dim_permut_tgt[:ndims], \
                                          num_data_elements_tgt[:ndims], \
                                          num_elements_tgt[:ndims], \
                                          num_ghost_tgt[:ndims], \
                                          lb_tgt[:ndims], \
                                          ub_tgt[:ndims], \
                                          lb_stripped_tgt[:ndims], \
                                          ub_stripped_tgt[:ndims])

    // Copy the class members related to the domain decomposition
    // from the source to the target op via OpenMP offload
      #pragma omp target teams default(none) defaultmap(none) \
              shared(dcomm_handler_src, dim_permut_tgt, \
                     num_data_elements_tgt, num_elements_tgt, \
                     num_ghost_tgt, lb_tgt, ub_tgt, \
                     lb_stripped_tgt, ub_stripped_tgt)
    {
        #pragma omp distribute simd
        for (int32_t n = 1; n <= dcomm_handler_src.get_n_dims(); n++)
        {
            dim_permut_tgt[n-1]        = dcomm_handler_src.get_dim_permut(n);
            num_data_elements_tgt[n-1] =
                dcomm_handler_src.get_num_data_elements(n);
            num_elements_tgt[n-1] = dcomm_handler_src.get_num_elements(n);
            num_ghost_tgt[n-1]    = dcomm_handler_src.get_num_ghosts(n);
            lb_tgt[n-1]           = dcomm_handler_src.get_lbound(n);
            ub_tgt[n-1]           = dcomm_handler_src.get_ubound(n);
            lb_stripped_tgt[n-1]  = dcomm_handler_src.get_lbound_stripped(n);
            ub_stripped_tgt[n-1]  = dcomm_handler_src.get_ubound_stripped(n);
        }
    }
    // Copy data from the device to the host
    #pragma omp target exit data map(from: dim_permut_tgt[:ndims], \
                                           num_data_elements_tgt[:ndims], \
                                           num_elements_tgt[:ndims], \
                                           num_ghost_tgt[:ndims], \
                                           lb_tgt[:ndims], \
                                           ub_tgt[:ndims], \
                                           lb_stripped_tgt[:ndims], \
                                           ub_stripped_tgt[:ndims])

    return 0;
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_dcomm_handler_domain_copy(
    const dcomm_handler_t** dcomm_handler_src_cxx_pptr,
    dcomm_handler_data_t* dcomm_handler_data_tgt)
{
    // Assign the source dcomm_handler_t class instance
    const dcomm_handler_t& dcomm_handler_src = *(*dcomm_handler_src_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return dcomm_handler_domain_copy_cxx(dcomm_handler_src,
                                                 dcomm_handler_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return dcomm_handler_domain_copy_acc(dcomm_handler_src,
                                                 dcomm_handler_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return dcomm_handler_domain_copy_ompx(dcomm_handler_src,
                                                  dcomm_handler_data_tgt);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
