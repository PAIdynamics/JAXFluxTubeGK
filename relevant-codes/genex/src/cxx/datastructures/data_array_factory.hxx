#ifndef DATA_ARRAY_FACTORY_HXX
#define DATA_ARRAY_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "data_array.hxx"
#include <array>
#include "data_array_omp.hxx"

#ifdef ENABLE_OPENACC
#include "data_array_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "data_array_ompx.hxx"
#endif

namespace data_array
{
    // Factory method to instantiate data_array_t derived class
    // based on the chosen GPU offload backend. This method is for
    // non-distributed non-interoperable data_array_t object defined
    // on C++ layer
    template<typename T, size_t DIM>
    data_array_t<T, DIM>* create(const std::array<int32_t, DIM>& lb,
                                 const std::array<int32_t, DIM>& ub,
                                 std::optional<T> init_value = std::nullopt)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new data_array_omp_t<T, DIM>(lb, ub, init_value);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new data_array_acc_t<T, DIM>(lb, ub, init_value);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new data_array_ompx_t<T, DIM>(lb, ub, init_value);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }

    // Factory method to instantiate data_array_t derived class
    // based on the chosen GPU offload backend. This method is for Fortran/C++
    // interoperable data_array_t object defined on Fortran layer.
    template<typename T, size_t DIM>
    data_array_t<T, DIM>* create(data_array_data_t* da_data)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new data_array_omp_t<T, DIM>(da_data);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new data_array_acc_t<T, DIM>(da_data);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new data_array_ompx_t<T, DIM>(da_data);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
