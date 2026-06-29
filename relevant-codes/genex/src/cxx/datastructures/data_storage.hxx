#ifndef DATA_STORAGE_HXX
#define DATA_STORAGE_HXX

#include "genex_cxx_env.hxx"
#include "dcomm_handler.hxx"
#include "data_array.hxx"
#include "mailbox.hxx"
#include "mailbox_factory.hxx"
#include "profiler.hxx"
#include <cassert>
#include <unordered_map>
#include <array>

// Error flag for data_storage_t
namespace data_storage_gpu
{
    inline bool is_erroneous = false;
}

#ifdef __cplusplus
extern "C" {
#endif

struct data_storage_data_t
{
    int32_t array_dim;
    int32_t n_ex_dims;
    int32_t* number_of_elements_ptr;
    int32_t* number_of_ghost_cells_ptr;
    int32_t* number_of_data_cells_ptr;
    int32_t* number_of_mail_partners_ptr;
    int32_t* dim_permut_ptr;
};

#ifdef __cplusplus
}
#endif

// Abstract class of C++ data_storage_gpu_t without template
class data_storage_gpu_base_t
{
public:
    // Default constructor
    data_storage_gpu_base_t() = default;

    // Virtual destructor
    virtual ~data_storage_gpu_base_t() = default;

    // Copy constructor is disabled.
    data_storage_gpu_base_t(const data_storage_gpu_base_t&) = delete;

    // Copy-assignment operator is disabled.
    data_storage_gpu_base_t& operator=(const data_storage_gpu_base_t&) = delete;

    // Pure virtual routine to copy the array from GPU to CPU
    virtual void update_host() = 0;

    // Pure virtual routine to copy the array from CPU to GPU
    virtual void update_device() = 0;
};

// C++ class which corresponds to the Fortran class data_storage_t with T as
// the type and DIM as the dimension of the array
// NOTE: data_storage_t object is not expected to be accessed from within
//       GPU compute region hence class members are not copied to GPU memory
template<typename T, size_t DIM>
class data_storage_gpu_t: public data_storage_gpu_base_t
{
protected:
    // Total number of elements of the array
    int64_t size;
    // Total number of elements of the array without ghost
    int64_t size_stripped;
    // Number of exchange dimension
    int32_t n_ex_dims;
    // Array specifying the number of elements for each dimension
    std::array<int32_t, DIM> shape;
    // Array specifying the number of elements without ghost for each dimension
    std::array<int32_t, DIM> shape_stripped;
    // Array specifying the lower boundary index for each dimension
    std::array<int32_t, DIM> lb;
    // Array specifying the upper boundary index for each dimension
    std::array<int32_t, DIM> ub;
    // Array specifying the lower boundary index without ghost
    // for each dimension
    std::array<int32_t, DIM> lb_stripped;
    // Array specifying the upper boundary index without ghost
    // for each dimension
    std::array<int32_t, DIM> ub_stripped;
    // Array specifying the number of dimensions of the stored data
    std::array<int32_t, DIM> number_of_elements;
    // Array specifying the number of ghost cells
    std::array<int32_t, DIM> number_of_ghost_cells;
    // Array specifying the number of data cells
    std::array<int32_t, DIM> number_of_data_cells;
    // Array specifying the number of mail partners for mpi ghost exchange
    std::array<int32_t, DIM> number_of_mail_partners;
    // Array specifying the order of the dimensions in storage.
    std::array<int32_t, DIM> dim_permut;

    // Map to mailboxes buffering data for boundary exchange
    std::unordered_map<int32_t, mailbox_t*> mailboxes;
    // Instance of the data array
    data_array_t<T, DIM>& data_array;
    // Communication handler
    const dcomm_handler_t& dcomm_handler;

public:
    // Constructor
    data_storage_gpu_t(const struct data_storage_data_t* ds_data,
                       data_array_t<T, DIM>** da_cxx_pptr,
                       const dcomm_handler_t** dcomm_handler_cxx_pptr)
    : data_storage_gpu_base_t(),
      data_array(*(*da_cxx_pptr)),
      dcomm_handler(*(*dcomm_handler_cxx_pptr))
    {
        // Assert if the given data is 2D or 4D or 5D
        static_assert(DIM == 2 || DIM == 4 || DIM == 5);

        this->size          = this->data_array.get_size();
        this->size_stripped = this->data_array.get_size_stripped();
        this->n_ex_dims     = ds_data->n_ex_dims;

        for (size_t i = 0; i < DIM; i++)
        {
            this->shape[i]          = this->data_array.get_shape(i+1);
            this->shape_stripped[i] = this->data_array.get_shape_stripped(i+1);
            this->lb[i]             = this->data_array.get_lbound(i+1);
            this->ub[i]             = this->data_array.get_ubound(i+1);
            this->lb_stripped[i]    = this->data_array.get_lbound_stripped(i+1);
            this->ub_stripped[i]    = this->data_array.get_ubound_stripped(i+1);

            this->number_of_elements[i]    = ds_data->number_of_elements_ptr[i];
            this->number_of_ghost_cells[i] =
                ds_data->number_of_ghost_cells_ptr[i];
            this->number_of_data_cells[i] =
                ds_data->number_of_data_cells_ptr[i];
            this->number_of_mail_partners[i] =
                ds_data->number_of_mail_partners_ptr[i];
            this->dim_permut[i] = ds_data->dim_permut_ptr[i];
        }

        // Initialize mailboxes for each exchange dimensions
        // Note: This has been generalized to work with 2D, 4D and 5D
        //       data storage
        for (int32_t i = 0; i < this->n_ex_dims; i++)
        {
            int32_t send_width;
            int32_t ex_dim = this->get_dim_permut(2 + i);

            if (this->get_num_mail_partners(ex_dim) == 2)
            {
                // Every mail partner gets the complete number of ghost cells
                send_width = this->get_num_ghost_cells(ex_dim);
            }
            else
            {
                // Every mail partner gets exactly one ghost cell
                send_width = 1;
            }

            int32_t n_mcells = send_width;
            for (int32_t j = 1; j <= DIM; j++)
            {
                if (j != ex_dim)
                {
                    n_mcells *= this->get_num_data_cells(j);
                }
            }
            int32_t n_mpartners = this->get_num_mail_partners(ex_dim);

            // Create mapping pairs between an exchange dimension index as key
            // and a pointer to mailbox object as value
            this->mailboxes.insert(
                std::make_pair(ex_dim, mailbox::create(n_mcells, n_mpartners)));

            data_storage_gpu::is_erroneous = data_storage_gpu::is_erroneous ||
                                             mailbox::is_erroneous;
        }
    }

    // Destructor
    ~data_storage_gpu_t()
    {
        // Iterate all entries within mailbox map
        for (auto& mb : this->mailboxes)
        {
            delete mb.second;

            data_storage_gpu::is_erroneous = data_storage_gpu::is_erroneous ||
                                             mailbox::is_erroneous;
        }
        this->mailboxes.clear();
    }

    // Copy constructor is disabled
    data_storage_gpu_t(const data_storage_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    data_storage_gpu_t& operator=(const data_storage_gpu_t&) = delete;

    // Routine to copy the array from GPU to CPU
    void update_host() override
    {
        this->data_array.update_host();
    }

    // Routine to copy the array from CPU to GPU
    void update_device() override
    {
        this->data_array.update_device();
    }

    // Getter for the number of dimension or rank of the array
    inline size_t get_dimension() const
    {
        return DIM;
    }

    // Getter for the total number of elements of the array
    inline int64_t get_size() const
    {
        return this->size;
    }

    // Getter the total number of elements of the array without ghost
    inline int64_t get_size_stripped() const
    {
        return this->size_stripped;
    }

    // Getter for the array specifying the number of elements for each dimension
    inline int32_t get_shape(int32_t i) const
    {
        return this->shape[i - 1];
    }

    // Getter for the the array specifying the number of elements without ghost
    // for each dimension
    inline int32_t get_shape_stripped(int32_t i) const
    {
        return this->shape_stripped[i - 1];
    }

    // Getter for the array specifying the lower boundary index for
    // each dimension
    inline int32_t get_lbound(int32_t i) const
    {
        return this->lb[i - 1];
    }

    // Getter for the array specifying the upper boundary index for
    // each dimension
    inline int32_t get_ubound(int32_t i) const
    {
        return this->ub[i - 1];
    }

    // Getter for the array specifying the lower boundary index
    // without ghost for each dimension
    inline int32_t get_lbound_stripped(int32_t i) const
    {
        return this->lb_stripped[i - 1];
    }

    // Getter for the array specifying the upper boundary index
    // without ghost for each dimension
    inline int32_t get_ubound_stripped(int32_t i) const
    {
        return this->ub_stripped[i - 1];
    }

    // Getter for the array specifying the number of elements of the stored data
    inline int32_t get_num_elements(int32_t i) const
    {
        return this->number_of_elements[i - 1];
    }

    // Getter for the array specifying the number of ghost cells
    inline int32_t get_num_ghost_cells(int32_t i) const
    {
        return this->number_of_ghost_cells[i - 1];
    }

    // Getter for the array specifying the number of data cells
    inline int32_t get_num_data_cells(int32_t i) const
    {
        return this->number_of_data_cells[i - 1];
    }

    // Getter for the array specifying of the number of mail partners
    // for mpi ghost exchange
    inline int32_t get_num_mail_partners(int32_t i) const
    {
        return this->number_of_mail_partners[i - 1];
    }

    // Getter for the array specifying the order of the dimensions in storage
    inline int32_t get_dim_permut(int32_t i) const
    {
        return this->dim_permut[i - 1];
    }

    // Getter for the number of echange dimension
    inline int32_t get_num_ex_dims() const
    {
        return this->n_ex_dims;
    }

    // Getter for the physical dimension associated to ex_dim via dim_permut
    // This corresponds to the inverse of dim_permut
    inline int32_t get_physical_dimension(int32_t ex_dim) const
    {
        // phi
        if(ex_dim == this->get_dim_permut(2))
        {
            return 2;
        }
        // mu
        else if(ex_dim == this->get_dim_permut(3))
        {
            return 3;
        }
        // vp
        else if(ex_dim == this->get_dim_permut(4))
        {
            return 4;
        }
        else
        {
            // Error in this case has been caught by check_ex_dim()
            return 0;
        }
    }

    // Start exchange of the ghost cells via mpi. This subroutine
    // returns immediately. Other calculations can be performed while
    // the exchange is ongoing.
    // Note: This has been generalized to work with 2D, 4D and 5D data storage
    void start_exchange(const int32_t ex_dim)
    {
        const std::string dim_str = std::to_string(DIM);
        const std::string pack_region_name = "pack_" + dim_str + "d";
        const std::string send_region_name = "send_" + dim_str + "d";

        std::array<int32_t, DIM> lb_pack;
        std::array<int32_t, DIM> ub_pack;
        mailbox_t& mailbox = *(this->mailboxes[ex_dim]);

        mailbox.allocate_device();

        profiler::start_region(pack_region_name);

        // Initialize the lower and upper bounds for the data to pack
        for (int32_t i = 1; i <= DIM; i++)
        {
            lb_pack[i-1] = this->get_lbound_stripped(i);
            ub_pack[i-1] = this->get_ubound_stripped(i);
        }

        // Pack (case #1: 4 mail partners, case #2: 2 mail partners)
        if (this->get_num_mail_partners(ex_dim) == 4)
        {
            // Set specific bounds for the exchange dimension in the case we
            // have four mail partners
            lb_pack[ex_dim-1] = this->get_lbound_stripped(ex_dim);
            ub_pack[ex_dim-1] = this->get_lbound_stripped(ex_dim);
            for (int32_t i = 1; i <= 4; i++)
            {
                mailbox.pack(lb_pack, ub_pack, this->data_array, i);
            }
        }
        else
        {
            // Left neighbor
            lb_pack[ex_dim-1] = this->get_lbound_stripped(ex_dim);
            ub_pack[ex_dim-1] = this->get_lbound_stripped(ex_dim)
                              + this->get_num_ghost_cells(ex_dim) - 1;
            mailbox.pack(lb_pack, ub_pack, this->data_array, 1);
            // Right neighbor
            lb_pack[ex_dim-1] = this->get_ubound_stripped(ex_dim)
                              - this->get_num_ghost_cells(ex_dim) + 1;
            ub_pack[ex_dim-1] = this->get_ubound_stripped(ex_dim);
            mailbox.pack(lb_pack, ub_pack, this->data_array, 2);
        }

        profiler::end_region(pack_region_name);

        profiler::start_region(send_region_name);

        mailbox.deliver_outboxes(this->dcomm_handler,
                                 this->get_physical_dimension(ex_dim),
                                 this->get_num_mail_partners(ex_dim));

        profiler::end_region(send_region_name);
    }

    // Finish exchange of the ghost cells via mpi. After this
    // subroutine has returned it is guaranteed that the information
    // of the ghost cells is present in the storage.
    // Note: This has been generalized to work with 2D, 4D and 5D data storage
    void finish_exchange(const int32_t ex_dim)
    {
        const std::string dim_str = std::to_string(DIM);
        const std::string recv_region_name = "receive_" + dim_str + "d";
        const std::string unpack_region_name = "unpack_" + dim_str + "d";

        std::array<int32_t, DIM> lb_pack;
        std::array<int32_t, DIM> ub_pack;
        std::array<int32_t, 4> ghost_indices;
        mailbox_t& mailbox = *(this->mailboxes[ex_dim]);

        profiler::start_region(recv_region_name);

        mailbox.finish_delivery(this->get_num_mail_partners(ex_dim));

        profiler::end_region(recv_region_name);

        profiler::start_region(unpack_region_name);

        // Initialize the lower and upper bounds for the data to unpack
        for (int32_t i = 1; i <= DIM; i++)
        {
            lb_pack[i-1] = this->get_lbound_stripped(i);
            ub_pack[i-1] = this->get_ubound_stripped(i);
        }

        // Unpack (case #1: 4 mail partners, case #2: 2 mail partners)
        if (this->get_num_mail_partners(ex_dim) == 4)
        {
            ghost_indices[0] = this->get_lbound(ex_dim);
            ghost_indices[1] = this->get_lbound_stripped(ex_dim) - 1;
            ghost_indices[2] = this->get_ubound_stripped(ex_dim) + 1;
            ghost_indices[3] = this->get_ubound(ex_dim);

            for (int32_t i = 1; i <= 4; i++)
            {
                // Unpack if the mail partner exists
                if (mailbox.get_partners(i) != MPI_PROC_NULL)
                {
                    lb_pack[ex_dim-1] = ghost_indices[i-1];
                    ub_pack[ex_dim-1] = ghost_indices[i-1];
                    mailbox.unpack(lb_pack, ub_pack, this->data_array, i);
                }
            }
        }
        else
        {
            // Unpack if the left mail partner exists
            if (mailbox.get_partners(1) != MPI_PROC_NULL)
            {
                lb_pack[ex_dim-1] = this->get_lbound(ex_dim);
                ub_pack[ex_dim-1] = this->get_lbound_stripped(ex_dim) - 1;
                mailbox.unpack(lb_pack, ub_pack, this->data_array, 1);
            }
            // Unpack if the right mail partner exists
            if (mailbox.get_partners(2) != MPI_PROC_NULL)
            {
                lb_pack[ex_dim-1] = this->get_ubound_stripped(ex_dim) + 1;
                ub_pack[ex_dim-1] = this->get_ubound(ex_dim);
                mailbox.unpack(lb_pack, ub_pack, this->data_array, 2);
            }
        }
        profiler::end_region(unpack_region_name);

        mailbox.deallocate_device();
    }
};

// Check if the ghost exchange dimension ex_dim is supported
template<typename T, size_t DIM>
bool is_ex_dim_supported(const data_storage_gpu_t<T, DIM>& ds,
                         const int32_t ex_dim)
{
    const int32_t phi_dim = ds.get_dim_permut(2);
    const int32_t mu_dim  = ds.get_dim_permut(3);
    const int32_t vp_dim  = ds.get_dim_permut(4);

    if((ex_dim != phi_dim) && (ex_dim != mu_dim) && (ex_dim != vp_dim))
    {
        std::cerr << "C++ error: The ghost exchange dimension = "
                  << ex_dim
                  << " is not supported!"
                  << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_data_storage_2d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 2>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr);

int32_t cbind_data_storage_4d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 4>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr);

int32_t cbind_data_storage_5d_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const struct data_storage_data_t* ds_data,
    data_array_t<real_t, 5>** da_cxx_pptr,
    data_storage_gpu_base_t** ds_cxx_pptr);

int32_t cbind_data_storage_2d_start_exchange(
    data_storage_gpu_t<real_t, 2>** ds_cxx_pptr);

int32_t cbind_data_storage_2d_finish_exchange(
    data_storage_gpu_t<real_t, 2>** ds_cxx_pptr);

int32_t cbind_data_storage_4d_start_exchange(
    data_storage_gpu_t<real_t, 4>** ds_cxx_pptr);

int32_t cbind_data_storage_4d_finish_exchange(
    data_storage_gpu_t<real_t, 4>** ds_cxx_pptr);

int32_t cbind_data_storage_5d_start_exchange(
    data_storage_gpu_t<real_t, 5>** ds_cxx_pptr, const int32_t ex_dim);

int32_t cbind_data_storage_5d_finish_exchange(
    data_storage_gpu_t<real_t, 5>** ds_cxx_pptr, const int32_t ex_dim);

int32_t cbind_data_storage_finalize(data_storage_gpu_base_t** ds_cxx_pptr);

int32_t cbind_data_storage_update_host(data_storage_gpu_base_t** da_cxx_pptr);

int32_t cbind_data_storage_update_device(data_storage_gpu_base_t** da_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
