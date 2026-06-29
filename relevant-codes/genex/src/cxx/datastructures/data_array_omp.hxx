#ifndef DATA_ARRAY_OMP_HXX
#define DATA_ARRAY_OMP_HXX

#include "genex_cxx_env.hxx"
#include "data_array.hxx"

// C++ class which corresponds to the Fortran class data_array_t on CPU
template<typename T, size_t DIM>
class data_array_omp_t: public data_array_t<T, DIM>
{
public:
    // OpenMP constructor for non-distributed non-interoperable data_array_t
    // object defined on C++ layer
    data_array_omp_t(const std::array<int32_t, DIM>& lb_in,
                     const std::array<int32_t, DIM>& ub_in,
                     std::optional<T> init_value = std::nullopt)
    : data_array_t<T, DIM>{lb_in, ub_in, init_value} {
        if (init_value)
        {
            op_set_uniform_omp_t op;
            op.apply(this->get_size(), init_value.value(), this->array);
        }
    };

    // OpenMP constructor for Fortran/C++ interoperable data_array_t object
    // defined on Fortran layer
    data_array_omp_t(struct data_array_data_t* data_array_data)
    : data_array_t<T, DIM>{data_array_data} {}

    // Destructor of the OpenMP child class
    ~data_array_omp_t() override {}

    // Copy constructor is disabled
    data_array_omp_t(const data_array_omp_t&) = delete;

    // Copy-assignment operator is disabled
    data_array_omp_t& operator=(const data_array_omp_t&) = delete;

    // Routine to copy the array from GPU to CPU. This does nothing.
    void update_host() override {}

    // Routine to copy the array from CPU to GPU. This does nothing
    void update_device() override {}

    // Routine to get the device pointer of the array via OpenMP
    // NOTE: In this case, it returns the host pointer of the array instead
    inline void* get_array_device_ptr() const override
    {
        return this->array;
    }
};

#endif
