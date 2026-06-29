#ifndef DCOMM_HANDLER_HXX
#define DCOMM_HANDLER_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include <unordered_map>
#include <mpi.h>
#include <cassert>

#ifdef ENABLE_COMM_NCCL
#ifdef __HIP_PLATFORM_AMD__
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif
#ifdef DOUBLE_PREC
const ncclDataType_t NCCL_REAL_T = ncclDouble;
#else
const ncclDataType_t NCCL_REAL_T = nnclFloat;
#endif

#endif

// Alias for external namespace
namespace dmd = device_memory_debugger;

// Error flag for dcomm_handler_t
namespace dcomm_handler
{
    inline bool is_erroneous = false;
}

#ifdef __cplusplus
extern "C" {
#endif
// Struct container from class members from Fortran
struct dcomm_handler_data_t
{
    // Communication part
    int32_t comm_base;
    int32_t comm_cart;
    int32_t comm_phi;
    int32_t comm_vp;
    int32_t comm_mu;
    int32_t comm_sp;
    int32_t comm_phi_sp;
    int32_t comm_vp_mu;
    int32_t comm_phi_vp_mu;
    int32_t comm_vp_mu_sp;
    int32_t comm_mu_sp;
    int32_t comm_phi_mu_sp;
    int32_t n_procs_total;
    int32_t n_procs_RZ;
    int32_t n_procs_phi;
    int32_t n_procs_vp;
    int32_t n_procs_mu;
    int32_t n_procs_sp;
    int32_t rank;

    // Domain decomposition part
    int32_t n_dims;
    int32_t* dim_permut_ptr;
    int32_t* number_of_data_elements_ptr;
    int32_t* number_of_elements_ptr;
    int32_t* number_of_ghosts_ptr;
    int32_t* lb_ptr;
    int32_t* ub_ptr;
    int32_t* lb_stripped_ptr;
    int32_t* ub_stripped_ptr;
};

#ifdef __cplusplus
}
#endif

// C++ class which corresponds to the Fortran class dcomm_handler_t
class dcomm_handler_t
{
protected:
    // Communication part

    // Base MPI communicator where to create the topology from
    MPI_Comm comm_base;
    // MPI communicator for the complete topology
    MPI_Comm comm_cart;
    // MPI communicator for the phi direction
    MPI_Comm comm_phi;
    // MPI communicator for the vp direction
    MPI_Comm comm_vp;
    // MPI communicator for the mu direction
    MPI_Comm comm_mu;
    // MPI communicator for the sp direction
    MPI_Comm comm_sp;
    // MPI communicator for the phi and sp direction
    MPI_Comm comm_phi_sp;
    // MPI communicator for the vp and mu direction
    MPI_Comm comm_vp_mu;
    // MPI communicator for the phi, vp and mu direction
    MPI_Comm comm_phi_vp_mu;
    // MPI communicator for the vp, mu and sp direction
    MPI_Comm comm_vp_mu_sp;
    // MPI communicator for the mu and sp direction
    MPI_Comm comm_mu_sp;
    // MPI communicator for the phi, mu and sp direction
    MPI_Comm comm_phi_mu_sp;
    // Number of procs in total
    int32_t n_procs_total;
    // Number of procs in RZ direction
    int32_t n_procs_RZ;
    // Number of procs in phi direction
    int32_t n_procs_phi;
    // Number of procs in vp direction
    int32_t n_procs_vp;
    // Number of procs in mu direction
    int32_t n_procs_mu;
    // Number of procs in sp direction
    int32_t n_procs_sp;
    // Rank ID where the code is running on
    int32_t rank;
    // Status of whether this process is the master (rank-0)
    bool is_master_process;

    // Domain decomposition part

    // Number of MPI parallelized dimensions
    int32_t n_dims;
    // Pointer to the array specifying the order of the dimensions
    int32_t* dim_permut_ptr;
    // Pointer to the array specifying the number of data elements
    // for each dimension (excludes ghost cells)
    int32_t* number_of_data_elements_ptr;
    // Pointer to the array specifying the number of elements
    // for each dimension (includes ghost cells)
    int32_t* number_of_elements_ptr;
    // Pointer to the array specifying the number of ghost cells
    // for each dimension
    int32_t* number_of_ghosts_ptr;
    // Pointer to the array specifying the lower boundary index
    // of each dimension
    int32_t* lb_ptr;
    // Pointer to the array specifying the upper boundary index
    // of each dimension
    int32_t* ub_ptr;
    // Pointer to the array specifying the lower boundary index
    // without ghost of each dimension
    int32_t* lb_stripped_ptr;
    // Pointer to the array specifying the upper boundary index
    // without ghost of each dimension
    int32_t* ub_stripped_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;
        int32_t dim = this->n_dims;

        dmd::start_region("dcomm_handler_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->dim_permut_ptr, dim);
        err = err || dmd::is_invalid(this->number_of_data_elements_ptr, dim);
        err = err || dmd::is_invalid(this->number_of_elements_ptr, dim);
        err = err || dmd::is_invalid(this->number_of_ghosts_ptr, dim);
        err = err || dmd::is_invalid(this->lb_ptr, dim);
        err = err || dmd::is_invalid(this->ub_ptr, dim);
        err = err || dmd::is_invalid(this->lb_stripped_ptr, dim);
        err = err || dmd::is_invalid(this->ub_stripped_ptr, dim);
        dmd::end_region("dcomm_handler_t");

        dcomm_handler::is_erroneous = dcomm_handler::is_erroneous || err;
    }

#ifdef ENABLE_COMM_NCCL

    // NCCL communicators as counterparts of the MPI communicators above
    ncclComm_t ncomm_base;
    ncclComm_t ncomm_cart;
    ncclComm_t ncomm_phi;
    ncclComm_t ncomm_vp;
    ncclComm_t ncomm_mu;
    ncclComm_t ncomm_sp;
    ncclComm_t ncomm_phi_sp;
    ncclComm_t ncomm_vp_mu;
    ncclComm_t ncomm_phi_vp_mu;
    ncclComm_t ncomm_vp_mu_sp;
    ncclComm_t ncomm_mu_sp;
    ncclComm_t ncomm_phi_mu_sp;

    // Unordered map MPI communicator to its NCCL counterpart
    std::unordered_map<MPI_Comm, ncclComm_t> map_nccl_comm;

    // Unordered map MPI communicator to its NCCL counterpart
    const std::unordered_map<MPI_Datatype, ncclDataType_t> map_nccl_dtype =
        { {MPI_REAL_T, NCCL_REAL_T} };

    // Unordered map MPI communicator to its NCCL counterpart
    const std::unordered_map<MPI_Op, ncclRedOp_t> map_nccl_op =
        { {MPI_SUM, ncclSum} };

    void construct_mpi2nccl_comm(MPI_Comm& mpi_comm, ncclComm_t& nccl_comm)
    {
        ncclResult_t status;
        ncclUniqueId nccl_uid;
        int32_t rank_loc, size_loc;

        MPI_Comm_size(mpi_comm, &size_loc);
        MPI_Comm_rank(mpi_comm, &rank_loc);
        if (rank_loc == 0)
        {
            status = ncclGetUniqueId(&nccl_uid);
            assert((status == ncclSuccess)
                  && "Error C++: ncclGetUniqueId() failed!");
        }
        MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm);
        status = ncclCommInitRank(&nccl_comm, size_loc, nccl_uid, rank_loc);
        assert((status == ncclSuccess)
               && "Error C++: ncclCommInitRank() failed!");
    }

#endif

public:
    // Constructor
    dcomm_handler_t(const struct dcomm_handler_data_t* dcomm_handler_data)
    {
        // Communication part
        this->comm_base      = MPI_Comm_f2c(dcomm_handler_data->comm_base);
        this->comm_cart      = MPI_Comm_f2c(dcomm_handler_data->comm_cart);
        this->comm_phi       = MPI_Comm_f2c(dcomm_handler_data->comm_phi);
        this->comm_vp        = MPI_Comm_f2c(dcomm_handler_data->comm_vp);
        this->comm_mu        = MPI_Comm_f2c(dcomm_handler_data->comm_mu);
        this->comm_sp        = MPI_Comm_f2c(dcomm_handler_data->comm_sp);
        this->comm_phi_sp    = MPI_Comm_f2c(dcomm_handler_data->comm_phi_sp);
        this->comm_vp_mu     = MPI_Comm_f2c(dcomm_handler_data->comm_vp_mu);
        this->comm_phi_vp_mu = MPI_Comm_f2c(dcomm_handler_data->comm_phi_vp_mu);
        this->comm_vp_mu_sp  = MPI_Comm_f2c(dcomm_handler_data->comm_vp_mu_sp);
        this->comm_mu_sp     = MPI_Comm_f2c(dcomm_handler_data->comm_mu_sp);
        this->comm_phi_mu_sp = MPI_Comm_f2c(dcomm_handler_data->comm_phi_mu_sp);
        this->n_procs_total  = dcomm_handler_data->n_procs_total;
        this->n_procs_RZ     = dcomm_handler_data->n_procs_RZ;
        this->n_procs_phi    = dcomm_handler_data->n_procs_phi;
        this->n_procs_vp     = dcomm_handler_data->n_procs_vp;
        this->n_procs_mu     = dcomm_handler_data->n_procs_mu;
        this->n_procs_sp     = dcomm_handler_data->n_procs_sp;
        this->rank           = dcomm_handler_data->rank;
        // Check if this process is the master (rank-0)
        this->is_master_process = (this->rank == 0);

        // Domain decomposition part
        this->n_dims = dcomm_handler_data->n_dims;
        this->dim_permut_ptr = dcomm_handler_data->dim_permut_ptr;
        this->number_of_data_elements_ptr =
            dcomm_handler_data->number_of_data_elements_ptr;
        this->number_of_elements_ptr =
            dcomm_handler_data->number_of_elements_ptr;
        this->number_of_ghosts_ptr = dcomm_handler_data->number_of_ghosts_ptr;
        this->lb_ptr = dcomm_handler_data->lb_ptr;
        this->ub_ptr = dcomm_handler_data->ub_ptr;
        this->lb_stripped_ptr = dcomm_handler_data->lb_stripped_ptr;
        this->ub_stripped_ptr = dcomm_handler_data->ub_stripped_ptr;
    }

    // Destructor
    virtual ~dcomm_handler_t() = default;

    // Copy constructor is disabled
    dcomm_handler_t(const dcomm_handler_t&) = delete;

    // Copy-assignment operator is disabled
    dcomm_handler_t& operator=(const dcomm_handler_t&) = delete;

    // Pure virtual getter for the number of total GPUs available
    virtual inline int32_t get_n_devices() const = 0;

    // Pure virtual getter for the GPU rank ID where the code is running on
    virtual inline int32_t get_gpu_rank() const = 0;

    // Pure virtual routine to update the RZ domain from CPU to GPU
    virtual void update_device_RZ_domain() = 0;

    // Pure virtual method to perform MPI nonblocking send communication
    // Return MPI error code
    virtual inline int Isend(const real_t* buf, int count,
        MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
        MPI_Request* request) const = 0;

    // Pure virtual method to perform MPI nonblocking receive communication
    // Return MPI error code
    virtual inline int Irecv(real_t* buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request* request) const = 0;

    // Pure virtual method to perform blocking allreduce communication
    // Return 0 for success and 1 for error
    virtual inline int32_t Allreduce(const real_t* sendbuff, real_t* recvbuff,
                                     size_t count, MPI_Datatype datatype,
                                     MPI_Op op, MPI_Comm comm) const = 0;

    // Getter for the base MPI communicator where to create the topology from
    inline MPI_Comm get_comm_base() const
    {
        return this->comm_base;
    }
    // Getter for the MPI communicator for the complete topology
    inline MPI_Comm get_comm_cart() const
    {
        return this->comm_cart;
    }
    // Getter for the MPI communicator for the phi direction
    inline MPI_Comm get_comm_phi() const
    {
        return this->comm_phi;
    }
    // Getter for the MPI communicator for the vp direction
    inline MPI_Comm get_comm_vp() const
    {
        return this->comm_vp;
    }
    // Getter for the MPI communicator for the mu direction
    inline MPI_Comm get_comm_mu() const
    {
        return this->comm_mu;
    }
    // Getter for the MPI communicator for the sp direction
    inline MPI_Comm get_comm_sp() const
    {
        return this->comm_sp;
    }
    // Getter for the MPI communicator for the RZ, phi and sp directions
    inline MPI_Comm get_comm_RZ_phi_sp() const
    {
        return this->comm_phi_sp;
    }
    // Getter for the MPI communicator for the RZ and phi directions
    inline MPI_Comm get_comm_RZ_phi() const
    {
        return this->comm_phi;
    }
    // Getter for the MPI communicator for the vp and mu directions
    inline MPI_Comm get_comm_vp_mu() const
    {
        return this->comm_vp_mu;
    }
    // Getter for the MPI communicator for the phi, vp and mu directions
    inline MPI_Comm get_comm_phi_vp_mu() const
    {
        return this->comm_phi_vp_mu;
    }
    // Getter for the MPI communicator for the vp, mu and sp directions
    inline MPI_Comm get_comm_vp_mu_sp() const
    {
        return this->comm_vp_mu_sp;
    }
    // Getter for the MPI communicator for the mu and sp directions
    inline MPI_Comm get_comm_mu_sp() const
    {
        return this->comm_mu_sp;
    }
    // Getter for the MPI communicator for the phi, mu and sp directions
    inline MPI_Comm get_comm_phi_mu_sp() const
    {
        return this->comm_phi_mu_sp;
    }
    // Getter for the number of procs in total
    inline int32_t get_n_procs_total() const
    {
        return this->n_procs_total;
    }
    // Getter for the number of procs in RZ direction
    inline int32_t get_n_procs_RZ() const
    {
        return this->n_procs_RZ;
    }
    // Getter for the number of procs in phi direction
    inline int32_t get_n_procs_phi() const
    {
        return this->n_procs_phi;
    }
    // Getter for the number of procs in vp direction
    inline int32_t get_n_procs_vp() const
    {
        return this->n_procs_vp;
    }
    // Getter for the number of procs in mu direction
    inline int32_t get_n_procs_mu() const
    {
        return this->n_procs_mu;
    }
    // Getter for the number of procs in sp direction
    inline int32_t get_n_procs_sp() const
    {
        return this->n_procs_sp;
    }
    // Getter for the rank ID where the code is running on
    inline int32_t get_rank() const
    {
        return this->rank;
    }
    // Getter for the status of whether this process is the master (rank-0)
    inline bool is_master() const
    {
        return this->is_master_process;
    }
    // Getter for the number of MPI parallelized dimensions
    inline int32_t get_n_dims() const
    {
        return this->n_dims;
    }
    // Getter for the size of the distributed domain with ghosts
    inline int64_t get_size() const
    {
        int64_t size = 0;
        for (int32_t j = 0; j < this->n_dims; j++)
        {
            size += this->ub_ptr[j] - this->lb_ptr[j] + 1;
        }
        return size;
    }
    // Getter for the size of the distributed domain without ghosts
    inline int64_t get_size_stripped() const
    {
        int64_t size_s = 0;
        for (int32_t j = 0; j < this->n_dims; j++)
        {
            size_s += this->ub_stripped_ptr[j] - this->lb_stripped_ptr[j] + 1;
        }
        return size_s;
    }
    // Getter for the array specifying the order of the dimensions
    #pragma acc routine seq
    inline int32_t get_dim_permut(int32_t i) const
    {
        return this->dim_permut_ptr[i - 1];
    }
    // Getter for the array specifying the number of data elements
    // for each dimension (excludes ghost cells)
    #pragma acc routine seq
    inline int32_t get_num_data_elements(int32_t i) const
    {
        return this->number_of_data_elements_ptr[i - 1];
    }
    // Getter for the array specifying the number of elements
    // for each dimension (includes ghost cells)
    #pragma acc routine seq
    inline int32_t get_num_elements(int32_t i) const
    {
        return this->number_of_elements_ptr[i - 1];
    }
    // Getter for the array specifying the number of ghost cells
    // for each dimension
    #pragma acc routine seq
    inline int32_t get_num_ghosts(int32_t i) const
    {
        return this->number_of_ghosts_ptr[i - 1];
    }
    // Getter for the array pointer specifying the lower boundary index of
    // each dimension
    #pragma acc routine seq
    inline const int32_t* get_lbound() const
    {
        return this->lb_ptr;
    }
    // Getter for the array specifying the lower boundary index of
    // each dimension
    #pragma acc routine seq
    inline int32_t get_lbound(int32_t i) const
    {
        return this->lb_ptr[i - 1];
    }
    // Getter for the array pointer specifying the upper boundary index of
    // each dimension
    #pragma acc routine seq
    inline const int32_t* get_ubound() const
    {
        return this->ub_ptr;
    }
    // Getter for the array specifying the upper boundary index of
    // each dimension
    #pragma acc routine seq
    inline int32_t get_ubound(int32_t i) const
    {
        return this->ub_ptr[i - 1];
    }
    // Getter for the array pointer specifying the lower boundary index
    // without ghost of each dimension
    #pragma acc routine seq
    inline const int32_t* get_lbound_stripped() const
    {
        return this->lb_stripped_ptr;
    }
    // Getter for the array specifying the lower boundary index
    // without ghost of each dimension
    #pragma acc routine seq
    inline int32_t get_lbound_stripped(int32_t i) const
    {
        return this->lb_stripped_ptr[i - 1];
    }
    // Getter for the array pointer specifying the upper boundary index
    // without ghost of each dimension
    #pragma acc routine seq
    inline const int32_t* get_ubound_stripped() const
    {
        return this->ub_stripped_ptr;
    }
    // Getter for the array specifying the upper boundary index
    // without ghost of each dimension
    #pragma acc routine seq
    inline int32_t get_ubound_stripped(int32_t i) const
    {
        return this->ub_stripped_ptr[i - 1];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success
int32_t cbind_dcomm_handler_initialize(
    const struct dcomm_handler_data_t* dcomm_handler_data,
    dcomm_handler_t** dcomm_handler_cxx_pptr);

// Return 0 for success
int32_t cbind_dcomm_handler_finalize(dcomm_handler_t** dcomm_handler_cxx_pptr);

// Return 0 for success
int32_t cbind_dcomm_handler_update_device_RZ(
    dcomm_handler_t** dcomm_handler_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
