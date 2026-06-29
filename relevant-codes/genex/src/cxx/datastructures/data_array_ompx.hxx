#ifndef DATA_ARRAY_OMPX_HXX
#define DATA_ARRAY_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "data_array.hxx"
#include "op_set_uniform_ompx.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class data_array_t
// with OpenMP offload
template<typename T, size_t DIM>
class data_array_ompx_t: public data_array_t<T, DIM>
{
public:
    // OpenMP offload constructor for non-distributed non-interoperable
    // data_array_t object defined on C++ layer
    data_array_ompx_t(const std::array<int32_t, DIM>& lb_in,
                      const std::array<int32_t, DIM>& ub_in,
                      std::optional<T> init_value = std::nullopt)
    : data_array_t<T, DIM>{lb_in, ub_in, init_value}
    {
        int64_t size = this->get_size();
        // Allocate device pointer, map the host pointer to it
        #pragma omp target enter data \
            map(to: this[:1], \
                this->lb[:DIM], \
                this->ub[:DIM], \
                this->lb_stripped[:DIM], \
                this->ub_stripped[:DIM]), \
            map(alloc: this->array[:size])

        if (init_value)
        {
            op_set_uniform_ompx_t op;
            op.apply(this->get_size(), init_value.value(), this->array);
        }

        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // OpenMP offload constructor for Fortran/C++ interoperable data_array_t
    // object defined on Fortran layer
    data_array_ompx_t(struct data_array_data_t* data_array_data)
    : data_array_t<T, DIM>{data_array_data}
    {
        int64_t size = this->get_size();
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data map(to: this[:1], \
                                              this->strides[:DIM], \
                                              this->lb[:DIM], \
                                              this->ub[:DIM], \
                                              this->lb_stripped[:DIM], \
                                              this->ub_stripped[:DIM], \
                                              this->array[:size])

        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // Destructor of the OpenMP offload child class
    ~data_array_ompx_t() override
    {
        int64_t size = this->get_size();
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data map(delete: this->array[:size], \
                                                 this->ub_stripped[:DIM], \
                                                 this->lb_stripped[:DIM], \
                                                 this->ub[:DIM], \
                                                 this->lb[:DIM], \
                                                 this->strides[:DIM], \
                                                 this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Copy constructor is disabled
    data_array_ompx_t(const data_array_ompx_t&) = delete;

    // Copy-assignment operator is disabled
    data_array_ompx_t& operator=(const data_array_ompx_t&) = delete;

    // Routine to copy the array from GPU to CPU via OpenMP offload
    void update_host() override
    {
        int64_t size = this->get_size();
        #pragma omp target update from(this->array[:size])
    }

    // Routine to copy the array from CPU to GPU via OpenMP offload
    void update_device() override
    {
        int64_t size = this->get_size();
        #pragma omp target update to(this->array[:size])
    }

    // Routine to get the device pointer of the array via OpenMP offload
    inline void* get_array_device_ptr() const override
    {
        int32_t gpu_rank = omp_get_default_device();
        return omp_get_mapped_ptr(this->array, gpu_rank);
    }
};

#endif
