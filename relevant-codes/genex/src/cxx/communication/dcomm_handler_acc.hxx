#ifndef DCOMM_HANDLER_ACC_HXX
#define DCOMM_HANDLER_ACC_HXX

#include "genex_cxx_env.hxx"
#include "dimensions.hxx"
#include "dcomm_handler.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class dcomm_handler_t with OpenACC
class dcomm_handler_acc_t: public dcomm_handler_t
{
private:
    // Number of total GPUs available
    int32_t n_devices;
    // GPU rank ID where the code is running on
    int32_t gpu_rank;

public:
    // Constructor
    dcomm_handler_acc_t(const struct dcomm_handler_data_t* dcomm_handler_data)
    : dcomm_handler_t{dcomm_handler_data}
    {
        this->n_devices = acc_get_num_devices(acc_device_nvidia);
        #pragma acc set device_num(this->rank % this->n_devices) \
                        device_type(acc_device_nvidia)
        this->gpu_rank = acc_get_device_num(acc_device_nvidia);

        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma acc enter data \
            copyin(this[:1], \
                   this->n_dims, \
                   this->dim_permut_ptr[:this->n_dims], \
                   this->number_of_data_elements_ptr[:this->n_dims], \
                   this->number_of_elements_ptr[:this->n_dims], \
                   this->number_of_ghosts_ptr[:this->n_dims], \
                   this->lb_ptr[:this->n_dims], \
                   this->ub_ptr[:this->n_dims], \
                   this->lb_stripped_ptr[:this->n_dims], \
                   this->ub_stripped_ptr[:this->n_dims])

        this->dmem_debug(dmd::mode_t::ALLOC);

        dcomm_handler::is_erroneous = dcomm_handler::is_erroneous ||
            (this->gpu_rank != (this->rank % this->n_devices));

#ifdef ENABLE_COMM_NCCL
        // Construct NCCL communicators from MPI communicators
        this->construct_mpi2nccl_comm(this->comm_base  , this->ncomm_base);
        this->construct_mpi2nccl_comm(this->comm_cart  , this->ncomm_cart);
        this->construct_mpi2nccl_comm(this->comm_phi   , this->ncomm_phi);
        this->construct_mpi2nccl_comm(this->comm_vp    , this->ncomm_vp);
        this->construct_mpi2nccl_comm(this->comm_mu    , this->ncomm_mu);
        this->construct_mpi2nccl_comm(this->comm_sp    , this->ncomm_sp);
        this->construct_mpi2nccl_comm(this->comm_phi_sp, this->ncomm_phi_sp);
        this->construct_mpi2nccl_comm(this->comm_vp_mu , this->ncomm_vp_mu);
        this->construct_mpi2nccl_comm(
            this->comm_phi_vp_mu, this->ncomm_phi_vp_mu);
        this->construct_mpi2nccl_comm(
            this->comm_vp_mu_sp, this->ncomm_vp_mu_sp);
        this->construct_mpi2nccl_comm(this->comm_mu_sp, this->ncomm_mu_sp);
        this->construct_mpi2nccl_comm(
            this->comm_phi_mu_sp, this->ncomm_phi_mu_sp);

        // Initialize mapping for MPI communicator to its NCCL counterpart
        this->map_nccl_comm[this->comm_base]      = this->ncomm_base;
        this->map_nccl_comm[this->comm_cart]      = this->ncomm_cart;
        this->map_nccl_comm[this->comm_phi]       = this->ncomm_phi;
        this->map_nccl_comm[this->comm_vp]        = this->ncomm_vp;
        this->map_nccl_comm[this->comm_mu]        = this->ncomm_mu;
        this->map_nccl_comm[this->comm_sp]        = this->ncomm_sp;
        this->map_nccl_comm[this->comm_phi_sp]    = this->ncomm_phi_sp;
        this->map_nccl_comm[this->comm_vp_mu]     = this->ncomm_vp_mu;
        this->map_nccl_comm[this->comm_phi_vp_mu] = this->ncomm_phi_vp_mu;
        this->map_nccl_comm[this->comm_vp_mu_sp]  = this->ncomm_vp_mu_sp;
        this->map_nccl_comm[this->comm_mu_sp]     = this->ncomm_mu_sp;
        this->map_nccl_comm[this->comm_phi_mu_sp] = this->ncomm_phi_mu_sp;
#endif
    }

    // Destructor
    ~dcomm_handler_acc_t() override
    {
        // Deallocate the device pointers of the class members from the device
        #pragma acc exit data \
            delete(this->ub_stripped_ptr[:this->n_dims], \
                   this->lb_stripped_ptr[:this->n_dims], \
                   this->ub_ptr[:this->n_dims], \
                   this->lb_ptr[:this->n_dims], \
                   this->number_of_ghosts_ptr[:this->n_dims], \
                   this->number_of_elements_ptr[:this->n_dims], \
                   this->number_of_data_elements_ptr[:this->n_dims], \
                   this->dim_permut_ptr[:this->n_dims], \
                   this->n_dims, \
                   this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Copy constructor is disabled
    dcomm_handler_acc_t(const dcomm_handler_acc_t&) = delete;

    // Copy-assignment operator is disabled
    dcomm_handler_acc_t& operator=(const dcomm_handler_acc_t&) = delete;

    // Getter for the number of total GPUs available
    inline int32_t get_n_devices() const override
    {
        return this->n_devices;
    }

    // Getter for the GPU rank ID where the code is running on
    inline int32_t get_gpu_rank() const override
    {
        return this->gpu_rank;
    }

    // Routine to update the RZ domain from CPU to GPU via OpenACC
    void update_device_RZ_domain() override
    {
        #pragma acc update \
            device(this->number_of_data_elements_ptr[DIM_RZ-1:1], \
                   this->number_of_elements_ptr[DIM_RZ-1:1], \
                   this->number_of_ghosts_ptr[DIM_RZ-1:1], \
                   this->lb_ptr[DIM_RZ-1:1], \
                   this->ub_ptr[DIM_RZ-1:1], \
                   this->lb_stripped_ptr[DIM_RZ-1:1], \
                   this->ub_stripped_ptr[DIM_RZ-1:1])
    }

    // Wrapper method to perform MPI nonblocking send communication
    // between GPUs via OpenACC. Return MPI error code.
    inline int Isend(const real_t* buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm, MPI_Request* request) const override
    {
        int32_t ierr;
#ifdef ENABLE_COMM_MPI_GPU
        // Using GPU-enabled MPI communication (direct GPU-GPU communication)
        // Currently not supported
        #pragma acc host_data use_device(buf)
        {
            ierr = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
        }
#else
        // Using non GPU-enabled MPI communication
        // (GPU-CPU-CPU-GPU communication)
        #pragma acc update self(buf[:count])
        ierr = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
#endif
        return ierr;
    }

    // Wrapper method to perform MPI nonblocking receive communication
    // between GPUs via OpenACC. Return MPI error code.
    inline int Irecv(real_t* buf, int count, MPI_Datatype datatype, int source,
        int tag, MPI_Comm comm, MPI_Request* request) const override
    {
        int32_t ierr;
#ifdef ENABLE_COMM_MPI_GPU
        // Using GPU-enabled MPI communication (direct GPU-GPU communication)
        // Currently not supported
        #pragma acc host_data use_device(buf)
        {
            ierr = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
        }
#else
        // Using non GPU-enabled MPI communication
        // (GPU-CPU-CPU-GPU communication)
        // NOTE: CPU to GPU copy is performed after MPI_Wait
        ierr = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
#endif
        return ierr;
    }

    // Wrapper method to perform blocking allreduce communication between GPUs
    // via OpenACC. Return 0 for success and 1 for error.
    inline int32_t Allreduce(const real_t* sendbuff, real_t* recvbuff,
                             size_t count, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm) const override
    {
        int32_t ierr;
#ifdef ENABLE_COMM_NCCL
        // Using NCCL for direct GPU-GPU collective communication
        ncclResult_t nerr;
        #pragma acc host_data use_device(sendbuff, recvbuff)
        {
            nerr = ncclAllReduce(sendbuff, recvbuff, count,
                                 this->map_nccl_dtype.at(datatype),
                                 this->map_nccl_op.at(op),
                                 this->map_nccl_comm.at(comm), 0);
        }
        ierr = (int32_t) (nerr != ncclSuccess);
#elif ENABLE_COMM_MPI_GPU
        // Using GPU-enabled MPI communication (direct GPU-GPU communication)
        #pragma acc host_data use_device(sendbuff, recvbuff)
        {
            ierr = MPI_Allreduce(sendbuff, recvbuff, count, datatype, op, comm);
        }
        ierr = (int32_t) (ierr != MPI_SUCCESS);
#else
        // Using non GPU-enabled MPI communication
        // (GPU-CPU-CPU-GPU communication)
        #pragma acc update self(sendbuff[:count])
        ierr = MPI_Allreduce(sendbuff, recvbuff, count, datatype, op, comm);
        #pragma acc update device(recvbuff[:count])
        ierr = (int32_t) (ierr != MPI_SUCCESS);
#endif
        return ierr;
    }
};

#endif
