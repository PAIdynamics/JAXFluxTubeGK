#include <cassert>
#include "genex_cxx_env.hxx"
#include "data_array.hxx"
#include "data_array_factory.hxx"

int32_t cbind_data_array_initialize(struct data_array_data_t* da_data,
                                    data_array_base_t** da_cxx_pptr)
{
    size_t dim = da_data->array_dim;

    // Allocate and construct the C++ class instance of the data array
    switch(dim)
    {
        case 2:
            *da_cxx_pptr = data_array::create<real_t, 2>(da_data);
            break;
        case 3:
            *da_cxx_pptr = data_array::create<real_t, 3>(da_data);
            break;
        case 4:
            *da_cxx_pptr = data_array::create<real_t, 4>(da_data);
            break;
        case 5:
            *da_cxx_pptr = data_array::create<real_t, 5>(da_data);
            break;
        default:
            data_array::is_erroneous = true;
            break;
    }

    return data_array::is_erroneous;
}

int32_t cbind_data_array_finalize(data_array_base_t** da_cxx_pptr)
{
    // Assign the data array C++ class instance
    data_array_base_t& da = *(*da_cxx_pptr);

    // Deallocate the host data array class instance
    delete &da;

    return data_array::is_erroneous;
}

int32_t cbind_data_array_update_host(data_array_base_t** da_cxx_pptr)
{
    // Assign the data array C++ class instance
    data_array_base_t& da = *(*da_cxx_pptr);

    // Update array on CPU from GPU
    da.update_host();

    return data_array::is_erroneous;
}

int32_t cbind_data_array_update_device(data_array_base_t** da_cxx_pptr)
{
    // Assign the data array C++ class instance
    data_array_base_t& da = *(*da_cxx_pptr);

    // Update array on GPU from CPU
    da.update_device();

    return data_array::is_erroneous;
}

void* cbind_data_array_get_device_pointer(data_array_base_t** da_cxx_pptr)
{
    // Assign the data array C++ class instance
    data_array_base_t &da = *(*da_cxx_pptr);

    return da.get_array_device_ptr();
}
