#ifndef HELPERS_DATA_ARRAY_CXX
#define HELPERS_DATA_ARRAY_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "data_array.hxx"

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Copy the class members from the source da class to the target da struct
// via OpenMP on CPU
template<typename T, size_t DIM>
int32_t data_array_copy_cxx(const data_array_t<T, DIM>& da_src,
                            struct data_array_data_t* da_data_tgt)
{
    // Assign array member pointer to the target da
    int32_t* shape_tgt          = da_data_tgt->array_shape_ptr;
    int32_t* shape_stripped_tgt = da_data_tgt->array_shape_stripped_ptr;
    int32_t* lb_tgt             = da_data_tgt->array_lb_ptr;
    int32_t* ub_tgt             = da_data_tgt->array_ub_ptr;
    int32_t* lb_stripped_tgt    = da_data_tgt->array_lb_stripped_ptr;
    int32_t* ub_stripped_tgt    = da_data_tgt->array_ub_stripped_ptr;

    // Copy the class members from the source to the target da via OpenMP
    #pragma omp parallel default(none) \
                         shared(da_src, da_data_tgt) \
                         shared(shape_tgt, shape_stripped_tgt, lb_tgt, ub_tgt) \
                         shared(lb_stripped_tgt, ub_stripped_tgt)
    {
        da_data_tgt->array_dim            = da_src.get_dimension();
        da_data_tgt->array_size           = da_src.get_size();
        da_data_tgt->array_size_stripped  = da_src.get_size_stripped();
        da_data_tgt->is_distributed_array = (int32_t) da_src.is_distributed();

        // #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= da_src.get_dimension(); i++)
        {
            shape_tgt[i-1]          = da_src.get_shape(i);
            shape_stripped_tgt[i-1] = da_src.get_shape_stripped(i);
            lb_tgt[i-1]             = da_src.get_lbound(i);
            ub_tgt[i-1]             = da_src.get_ubound(i);
            lb_stripped_tgt[i-1]    = da_src.get_lbound_stripped(i);
            ub_stripped_tgt[i-1]    = da_src.get_ubound_stripped(i);
        }
    }

    // Return 0 for success
    return 0;
}

#ifdef ENABLE_OPENACC
// Copy the class members from the source da class to the target da struct
// via OpenACC on GPU
template<typename T, size_t DIM>
int32_t data_array_copy_acc(const data_array_t<T, DIM>& da_src,
                            struct data_array_data_t* da_data_tgt)
{
    int32_t dim = da_src.get_dimension();

    // Assign array member pointer to the target da
    int32_t dim_tgt[1]          = {};
    int64_t size_tgt[2]         = {};
    int32_t* shape_tgt          = da_data_tgt->array_shape_ptr;
    int32_t* shape_stripped_tgt = da_data_tgt->array_shape_stripped_ptr;
    int32_t* lb_tgt             = da_data_tgt->array_lb_ptr;
    int32_t* ub_tgt             = da_data_tgt->array_ub_ptr;
    int32_t* lb_stripped_tgt    = da_data_tgt->array_lb_stripped_ptr;
    int32_t* ub_stripped_tgt    = da_data_tgt->array_ub_stripped_ptr;

    // Allocate and copy data from the host to the device
    #pragma acc enter data create(dim_tgt[:1], \
                                  size_tgt[:2], \
                                  shape_tgt[:dim], \
                                  shape_stripped_tgt[:dim], \
                                  lb_tgt[:dim], \
                                  ub_tgt[:dim], \
                                  lb_stripped_tgt[:dim], \
                                  ub_stripped_tgt[:dim])

    // Copy the class members from the source to the target da via OpenACC
    #pragma acc parallel default(none) \
                         present(da_src, dim_tgt, size_tgt) \
                         present(shape_tgt, shape_stripped_tgt, lb_tgt) \
                         present(ub_tgt, lb_stripped_tgt, ub_stripped_tgt)
    {
        dim_tgt[0]  = da_src.get_dimension();
        size_tgt[0] = da_src.get_size();
        size_tgt[1] = da_src.get_size_stripped();

        #pragma acc loop independent
        for (int32_t i = 1; i <= da_src.get_dimension(); i++)
        {
            shape_tgt[i-1]          = da_src.get_shape(i);
            shape_stripped_tgt[i-1] = da_src.get_shape_stripped(i);
            lb_tgt[i-1]             = da_src.get_lbound(i);
            ub_tgt[i-1]             = da_src.get_ubound(i);
            lb_stripped_tgt[i-1]    = da_src.get_lbound_stripped(i);
            ub_stripped_tgt[i-1]    = da_src.get_ubound_stripped(i);
        }
    }

    // Copy the class members from the target to the source op via OpenACC
    #pragma acc exit data copyout(dim_tgt[:1], \
                                  size_tgt[:2], \
                                  shape_tgt[:dim], \
                                  shape_stripped_tgt[:dim], \
                                  lb_tgt[:dim], \
                                  ub_tgt[:dim], \
                                  lb_stripped_tgt[:dim], \
                                  ub_stripped_tgt[:dim])

    da_data_tgt->array_dim            = dim_tgt[0];
    da_data_tgt->array_size           = size_tgt[0];
    da_data_tgt->array_size_stripped  = size_tgt[1];
    da_data_tgt->is_distributed_array = (int32_t) da_src.is_distributed();

    return 0;
}
#endif

#ifdef ENABLE_OPENMPX
// Copy the class members from the source da class to the target da struct
// via OpenMP offload on GPU
template<typename T, size_t DIM>
int32_t data_array_copy_ompx(const data_array_t<T, DIM>& da_src,
                             struct data_array_data_t* da_data_tgt)
{
    int32_t dim = da_src.get_dimension();

    // Assign array member pointer to the target da
    int32_t dim_tgt[1]          = {};
    int64_t size_tgt[2]         = {};
    int32_t* shape_tgt          = da_data_tgt->array_shape_ptr;
    int32_t* shape_stripped_tgt = da_data_tgt->array_shape_stripped_ptr;
    int32_t* lb_tgt             = da_data_tgt->array_lb_ptr;
    int32_t* ub_tgt             = da_data_tgt->array_ub_ptr;
    int32_t* lb_stripped_tgt    = da_data_tgt->array_lb_stripped_ptr;
    int32_t* ub_stripped_tgt    = da_data_tgt->array_ub_stripped_ptr;

    // Allocate and copy data from the host to the device
    #pragma omp target enter data \
        map(alloc: dim_tgt[:1], \
                   size_tgt[:2], \
                   shape_tgt[:dim], \
                   shape_stripped_tgt[:dim], \
                   lb_tgt[:dim], \
                   ub_tgt[:dim], \
                   lb_stripped_tgt[:dim], \
                   ub_stripped_tgt[:dim])

    // Copy the class members from the source to the target da
    // via OpenMP offload
    #pragma omp target teams default(none) defaultmap(none) \
        shared(da_src, dim_tgt, size_tgt) \
        shared(shape_tgt, shape_stripped_tgt, lb_tgt) \
        shared(ub_tgt, lb_stripped_tgt, ub_stripped_tgt)
    {
        dim_tgt[0]  = da_src.get_dimension();
        size_tgt[0] = da_src.get_size();
        size_tgt[1] = da_src.get_size_stripped();

        #pragma omp distribute simd
        for (int32_t i = 1; i <= da_src.get_dimension(); i++)
        {
            shape_tgt[i-1]          = da_src.get_shape(i);
            shape_stripped_tgt[i-1] = da_src.get_shape_stripped(i);
            lb_tgt[i-1]             = da_src.get_lbound(i);
            ub_tgt[i-1]             = da_src.get_ubound(i);
            lb_stripped_tgt[i-1]    = da_src.get_lbound_stripped(i);
            ub_stripped_tgt[i-1]    = da_src.get_ubound_stripped(i);
        }
    }

    // Copy the class members from the target to the source op
    // via OpenMP offload
    #pragma omp target exit data map(from: dim_tgt[:1], \
                                           size_tgt[:2], \
                                           shape_tgt[:dim], \
                                           shape_stripped_tgt[:dim], \
                                           lb_tgt[:dim], \
                                           ub_tgt[:dim], \
                                           lb_stripped_tgt[:dim], \
                                           ub_stripped_tgt[:dim])

    da_data_tgt->array_dim            = dim_tgt[0];
    da_data_tgt->array_size           = size_tgt[0];
    da_data_tgt->array_size_stripped  = size_tgt[1];
    da_data_tgt->is_distributed_array = (int32_t) da_src.is_distributed();

    return 0;
}
#endif

template<typename T, size_t DIM>
int32_t data_array_copy_switch(const data_array_t<T, DIM>& da_src,
                               struct data_array_data_t* da_data_tgt)
{
    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return data_array_copy_cxx(da_src, da_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return data_array_copy_acc(da_src, da_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return data_array_copy_ompx(da_src, da_data_tgt);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

// Interoperable routine to copy the class members from the source da class
// to the target da struct for 2D case
int32_t cbind_data_array_2d_copy(
    const data_array_t<real_t, 2>** da_src_cxx_pptr,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_array_t<real_t, 2>& da_src = *(*da_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_array_copy_switch(da_src, da_data_tgt);
}

// Interoperable routine to copy the class members from the source da class
// to the target da struct for 3D case
int32_t cbind_data_array_3d_copy(
    const data_array_t<real_t, 3>** da_src_cxx_pptr,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_array_t<real_t, 3>& da_src = *(*da_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_array_copy_switch(da_src, da_data_tgt);
}

// Interoperable routine to copy the class members from the source da class
// to the target da struct for 4D case
int32_t cbind_data_array_4d_copy(
    const data_array_t<real_t, 4>** da_src_cxx_pptr,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_array_t<real_t, 4>& da_src = *(*da_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_array_copy_switch(da_src, da_data_tgt);
}

// Interoperable routine to copy the class members from the source da class
// to the target da struct for 5D case
int32_t cbind_data_array_5d_copy(
    const data_array_t<real_t, 5>** da_src_cxx_pptr,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_array_t<real_t, 5>& da_src = *(*da_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_array_copy_switch(da_src, da_data_tgt);
}

#ifdef __cplusplus
}
#endif

#endif
