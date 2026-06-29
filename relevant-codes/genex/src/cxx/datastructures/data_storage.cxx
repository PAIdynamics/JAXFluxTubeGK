#include "data_storage.hxx"

int32_t cbind_data_storage_2d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 2>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the data storage
    *ds_cxx_pptr = new data_storage_gpu_t<real_t, 2>(ds_data, da_cxx_pptr,
                                                     dcomm_handler_cxx_pptr);
    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_4d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 4>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the data storage
    *ds_cxx_pptr = new data_storage_gpu_t<real_t, 4>(ds_data, da_cxx_pptr,
                                                     dcomm_handler_cxx_pptr);
    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_5d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 5>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the data storage
    *ds_cxx_pptr = new data_storage_gpu_t<real_t, 5>(ds_data, da_cxx_pptr,
                                                     dcomm_handler_cxx_pptr);
    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_2d_start_exchange(
    data_storage_gpu_t<real_t, 2>** ds_cxx_pptr)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 2>& ds_2d = *(*ds_cxx_pptr);

    // data_storage_t 2D can only be exchanged in phi direction
    ds_2d.start_exchange(ds_2d.get_dim_permut(2));

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_2d_finish_exchange(
    data_storage_gpu_t<real_t, 2>** ds_cxx_pptr)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 2>& ds_2d = *(*ds_cxx_pptr);

    // data_storage_t 2D can only be exchanged in phi direction
    ds_2d.finish_exchange(ds_2d.get_dim_permut(2));

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_4d_start_exchange(
    data_storage_gpu_t<real_t, 4>** ds_cxx_pptr)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 4>& ds_4d = *(*ds_cxx_pptr);

    // data_storage_t 4D can only be exchanged in phi direction
    ds_4d.start_exchange(ds_4d.get_dim_permut(2));

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_4d_finish_exchange(
    data_storage_gpu_t<real_t, 4>** ds_cxx_pptr)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 4>& ds_4d = *(*ds_cxx_pptr);

    // data_storage_t 4D can only be exchanged in phi direction
    ds_4d.finish_exchange(ds_4d.get_dim_permut(2));

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_5d_start_exchange(
    data_storage_gpu_t<real_t, 5>** ds_cxx_pptr, const int32_t ex_dim)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 5>& ds_5d = *(*ds_cxx_pptr);

    if(!is_ex_dim_supported(ds_5d, ex_dim)) return 1; // Return error
    ds_5d.start_exchange(ex_dim);

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_5d_finish_exchange(
    data_storage_gpu_t<real_t, 5>** ds_cxx_pptr, const int32_t ex_dim)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_t<real_t, 5>& ds_5d = *(*ds_cxx_pptr);

    if(!is_ex_dim_supported(ds_5d, ex_dim)) return 1; // Return error
    ds_5d.finish_exchange(ex_dim);

    return (int32_t) data_storage_gpu::is_erroneous;
}

int32_t cbind_data_storage_finalize(data_storage_gpu_base_t** ds_cxx_pptr)
{
    // Assign the data storage C++ class instance
    data_storage_gpu_base_t& ds = *(*ds_cxx_pptr);

    // Deallocate the host data storage class instance
    delete &ds;

    return (int32_t) data_storage_gpu::is_erroneous;
}
