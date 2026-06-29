#ifndef DATA_ARRAY_HXX
#define DATA_ARRAY_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "op_set_uniform_omp.hxx"
#include "params_gpu_offload.hxx"
#include <cassert>
#include <type_traits>
#include <array>
#include <optional>

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pgpus = params_gpu_offload;

// Error flag for data_array_t
namespace data_array
{
    inline bool is_erroneous = false;
}

#ifdef __cplusplus
extern "C" {
#endif

struct data_array_data_t
{
    real_t init_value;
    int32_t array_dim;
    int64_t array_size;
    int64_t array_size_stripped;
    int32_t is_distributed_array;
    int32_t* array_shape_ptr;
    int32_t* array_shape_stripped_ptr;
    int32_t* array_lb_ptr;
    int32_t* array_ub_ptr;
    int32_t* array_lb_stripped_ptr;
    int32_t* array_ub_stripped_ptr;
    real_t* array_ptr;
};
#ifdef __cplusplus
}
#endif

// Abstract class of C++ data_array_t without template
class data_array_base_t
{
public:
    // Default constructor
    data_array_base_t() = default;

    // Virtual destructor
    virtual ~data_array_base_t() = default;

    // Copy constructor is disabled.
    data_array_base_t(const data_array_base_t&) = delete;

    // Copy-assignment operator is disabled.
    data_array_base_t& operator=(const data_array_base_t&) = delete;

    // Pure virtual routine to copy the array from GPU to CPU
    virtual void update_host() = 0;

    // Pure virtual routine to copy the array from CPU to GPU
    virtual void update_device() = 0;

    // Pure virtual routine to get the device pointer of the array
    // Note: This return host pointer if GPU offloading backend is not used
    virtual inline void* get_array_device_ptr() const = 0;
};

// C++ class which corresponds to the Fortran class data_array_t with T as
// the type and DIM as the dimension of the array
template<typename T, size_t DIM>
class data_array_t: public data_array_base_t
{
protected:
    // Local alias-declaration for bound-array type
    using bounds_t = int32_t[DIM];

    // Array specifying the memory strides of the elements for each dimension
    bounds_t strides;
    // Array specifying the lower boundary index for each dimension
    bounds_t lb;
    // Array specifying the upper boundary index for each dimension
    bounds_t ub;
    // Array specifying the lower boundary index without ghost
    // for each dimension
    bounds_t lb_stripped;
    // Array specifying the upper boundary index without ghost
    // for each dimension
    bounds_t ub_stripped;
    // Pointer to the multidimensional array stored in the data array type
    T* array;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("data_array_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->strides, DIM);
        err = err || dmd::is_invalid(this->lb, DIM);
        err = err || dmd::is_invalid(this->ub, DIM);
        err = err || dmd::is_invalid(this->lb_stripped, DIM);
        err = err || dmd::is_invalid(this->ub_stripped, DIM);
        err = err || dmd::is_invalid(this->array, this->get_size());
        dmd::end_region("data_array_t");

        data_array::is_erroneous = data_array::is_erroneous || err;
    }

public:
    // Constructor for non-distributed non-interoperable data_array_t object
    // defined on C++ layer
    // TODO: Replace with std::span if C++20 is used
    data_array_t(const std::array<int32_t, DIM>& lb_in,
                 const std::array<int32_t, DIM>& ub_in,
                 std::optional<T> init_value = std::nullopt)
    : data_array_base_t()
    {
        // Assert if the given data has the same dimensionality as prescribed
        // and if it is supported
        static_assert(2 <= DIM && DIM <= 5, "DIM has to be in [2,3,4,5]");

        this->strides[0] = ub_in[0] - lb_in[0] + 1;
        for (size_t i = 0; i < DIM; i++)
        {
            this->lb[i]             = lb_in[i];
            this->ub[i]             = ub_in[i];
            this->lb_stripped[i]    = lb_in[i];
            this->ub_stripped[i]    = ub_in[i];
            if(i > 0)
            {
                this->strides[i] = this->strides[i - 1]
                                 * (ub_in[i] - lb_in[i] + 1);
            }
        }

        if(pgpus::get_use_array_alignment())
        {
            std::align_val_t align = pgpus::get_array_alignment();
            this->array = new (align) T[this->get_size()] {};
        }
        else
        {
            this->array = new T[this->get_size()] {};
        }
        /* Initialization is done in the child classes.*/
    }

    // Constructor for Fortran/C++ interoperable data_array_t object
    // defined on Fortran layer
    data_array_t(data_array_data_t* da_data)
    : data_array_base_t()
    {
        // Assert if the given data has the same dimensionality as prescribed
        // and if it is supported
        static_assert(2 <= DIM && DIM <= 5, "DIM has to be in [2,3,4,5]");
        assert(DIM == da_data->array_dim);

        this->strides[0] = da_data->array_shape_ptr[0];
        for (size_t i = 0; i < DIM; i++)
        {
            this->lb[i]             = da_data->array_lb_ptr[i];
            this->ub[i]             = da_data->array_ub_ptr[i];
            this->lb_stripped[i]    = da_data->array_lb_stripped_ptr[i];
            this->ub_stripped[i]    = da_data->array_ub_stripped_ptr[i];
            if (i > 0)
            {
                this->strides[i] = this->strides[i - 1]
                                 * da_data->array_shape_ptr[i];
            }
        }

        if(pgpus::get_use_array_alignment())
        {
            std::align_val_t align = pgpus::get_array_alignment();
            this->array = new (align) T[this->get_size()] {};
        }
        else
        {
            this->array = new T[this->get_size()] {};
        }

        op_set_uniform_omp_t op;
        op.apply(this->get_size(), da_data->init_value, this->array);
        da_data->array_ptr = this->array;
    }

    // Destructor
    virtual ~data_array_t()
    {
        if (pgpus::get_use_array_alignment())
        {
            operator delete(this->array, pgpus::get_array_alignment());
        }
        else
        {
            delete[] this->array;
        }
    }

    // Copy constructor is disabled
    data_array_t(const data_array_t&) = delete;

    // Copy-assignment operator is disabled
    data_array_t& operator=(const data_array_t&) = delete;

    // Getter for the number of dimension or rank of the array
    #pragma acc routine seq
    inline size_t get_dimension() const
    {
        return DIM;
    }

    // Getter for the total number of elements of the array
    #pragma acc routine seq
    inline int64_t get_size() const
    {
        return this->strides[DIM - 1];
    }

    // Getter the total number of elements of the array without ghost
    #pragma acc routine seq
    inline int64_t get_size_stripped() const
    {
        int64_t size_stripped = 1;
        for (size_t i = 0; i < DIM; i++)
        {
            size_stripped *= (this->ub_stripped[i] - this->lb_stripped[i] + 1);
        }
        return size_stripped;
    }

    // Getter for the array specifying the number of elements for each dimension
    #pragma acc routine seq
    inline int32_t get_shape(int32_t i) const
    {
        return this->ub[i - 1] - this->lb[i - 1] + 1;
    }

    // Getter for the the array specifying the number of elements without ghost
    // for each dimension
    #pragma acc routine seq
    inline int32_t get_shape_stripped(int32_t i) const
    {
        return this->ub_stripped[i - 1] - this->lb_stripped[i - 1] + 1;
    }

    // Getter for the array pointer specifying the lower boundary index for
    // each dimension
    #pragma acc routine seq
    inline const bounds_t& get_lbound() const
    {
        return this->lb;
    }

    // Getter for the array specifying the lower boundary index for
    // each dimension
    #pragma acc routine seq
    inline int32_t get_lbound(int32_t i) const
    {
        return this->lb[i - 1];
    }

    // Getter for the array pointer specifying the upper boundary index for
    // each dimension
    #pragma acc routine seq
    inline const bounds_t& get_ubound() const
    {
        return this->ub;
    }

    // Getter for the array specifying the upper boundary index for
    // each dimension
    #pragma acc routine seq
    inline int32_t get_ubound(int32_t i) const
    {
        return this->ub[i - 1];
    }

    // Getter for the array pointer specifying the lower boundary index
    // without ghost for each dimension
    #pragma acc routine seq
    inline const bounds_t& get_lbound_stripped() const
    {
        return this->lb_stripped;
    }

    // Getter for the array specifying the lower boundary index
    // without ghost for each dimension
    #pragma acc routine seq
    inline int32_t get_lbound_stripped(int32_t i) const
    {
        return this->lb_stripped[i - 1];
    }

    // Getter for the array pointer specifying the upper boundary index
    // without ghost for each dimension
    #pragma acc routine seq
    inline const bounds_t& get_ubound_stripped() const
    {
        return this->ub_stripped;
    }

    // Getter for the array specifying the upper boundary index
    // without ghost for each dimension
    #pragma acc routine seq
    inline int32_t get_ubound_stripped(int32_t i) const
    {
        return this->ub_stripped[i - 1];
    }

    // Returns true if the array is distributed array, otherwise false
    #pragma acc routine seq
    inline bool is_distributed() const
    {
        bool eq_bounds = true;
        for (size_t d = 0; d < DIM; d++)
        {
            eq_bounds = eq_bounds && (this->lb[d] == this->lb_stripped[d]);
            eq_bounds = eq_bounds && (this->ub[d] == this->ub_stripped[d]);
        }
        return !eq_bounds;
    }

    // Getter for 2D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline const typename std::enable_if<N == 2, T>::type&
    operator()(int32_t j0, int32_t j1) const
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]];
    }

    // Setter for 2D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline typename std::enable_if<N == 2, T>::type&
    operator()(int32_t j0, int32_t j1)
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]];
    }

    // Getter for 3D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline const typename std::enable_if<N == 3, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2) const
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]];
    }

    // Setter for 3D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline typename std::enable_if<N == 3, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2)
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]];
    }

    // Getter for 4D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline const typename std::enable_if<N == 4, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2, int32_t j3) const
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]
                                        + (j3 - lb[3]) * strides[2]];
    }

    // Setter for 4D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline typename std::enable_if<N == 4, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2, int32_t j3)
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]
                                        + (j3 - lb[3]) * strides[2]];
    }

    // Getter for 5D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline const typename std::enable_if<N == 5, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2, int32_t j3, int32_t j4) const
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]
                                        + (j3 - lb[3]) * strides[2]
                                        + (j4 - lb[4]) * strides[3]];
    }

    // Setter for 5D array
    #pragma acc routine seq
    template<size_t N = DIM>
    inline typename std::enable_if<N == 5, T>::type&
    operator()(int32_t j0, int32_t j1, int32_t j2, int32_t j3, int32_t j4)
    {
        return this->array[(j0 - lb[0]) + (j1 - lb[1]) * strides[0]
                                        + (j2 - lb[2]) * strides[1]
                                        + (j3 - lb[3]) * strides[2]
                                        + (j4 - lb[4]) * strides[3]];
    }

    // Getter for the pointer of the array
    #pragma acc routine seq
    inline T* get_array_ptr()
    {
        return this->array;
    }

};

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_data_array_initialize(struct data_array_data_t* da_data,
                                    data_array_base_t** da_cxx_pptr);

int32_t cbind_data_array_finalize(data_array_base_t** da_cxx_pptr);

int32_t cbind_data_array_update_host(data_array_base_t** da_cxx_pptr);

int32_t cbind_data_array_update_device(data_array_base_t** da_cxx_pptr);

void* cbind_data_array_get_device_pointer(data_array_base_t** da_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
