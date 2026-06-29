#include "dcomm_handler.hxx"
#include "dcomm_handler_factory.hxx"

int32_t cbind_dcomm_handler_initialize(
    const dcomm_handler_data_t* dcomm_handler_data,
    dcomm_handler_t** dcomm_handler_cxx_pptr)
{
    // Allocate and construct dcomm_handler_t C++ class instance
    *dcomm_handler_cxx_pptr = dcomm_handler::create(dcomm_handler_data);

    return (int32_t) dcomm_handler::is_erroneous;
}

int32_t cbind_dcomm_handler_finalize(dcomm_handler_t** dcomm_handler_cxx_pptr)
{
    // Assign the dcomm_handler_t class instance
    dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);

    // Deallocate the host dcomm_handler_t C++ class instance
    delete &dcomm_handler;

    return (int32_t) dcomm_handler::is_erroneous;
}

int32_t cbind_dcomm_handler_update_device_RZ(
    dcomm_handler_t** dcomm_handler_cxx_pptr)
{
    // Assign the dcomm_handler_t class instance
    dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);

    // Update RZ domain on GPU from CPU
    dcomm_handler.update_device_RZ_domain();

    return dcomm_handler::is_erroneous;
}
