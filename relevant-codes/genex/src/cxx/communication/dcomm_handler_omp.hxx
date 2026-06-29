#ifndef DCOMM_HANDLER_OMP_HXX
#define DCOMM_HANDLER_OMP_HXX

#include "genex_cxx_env.hxx"
#include "dcomm_handler.hxx"
#include <mpi.h>
#include <cassert>

// C++ class which corresponds to the Fortran class dcomm_handler_t on CPU
class dcomm_handler_omp_t: public dcomm_handler_t
{
public:
    // Constructor
    dcomm_handler_omp_t(const struct dcomm_handler_data_t* dcomm_handler_data)
    : dcomm_handler_t{dcomm_handler_data}
    {
        int32_t mpi_status, rank_local;
        mpi_status = MPI_Comm_rank(this->comm_base, &rank_local);
        dcomm_handler::is_erroneous = ((mpi_status != MPI_SUCCESS) ||
                                       (rank_local != this->rank));
    }

    // Destructor
    ~dcomm_handler_omp_t() override = default;

    // Copy constructor is disabled
    dcomm_handler_omp_t(const dcomm_handler_omp_t&) = delete;

    // Copy-assignment operator is disabled
    dcomm_handler_omp_t& operator=(const dcomm_handler_omp_t&) = delete;

    // Getter for the number of total GPUs available. It should not be called.
    inline int32_t get_n_devices() const override
    {
        assert(false && "Error C++: Method get_n_devices() is not callable.");
        return -1;
    }

    // Getter for the GPU rank ID where the code is running on
    // It should not be called.
    inline int32_t get_gpu_rank() const override
    {
        assert(false && "Error C++: Method get_gpu_rank() is not callable.");
        return -1;
    }

    // Routine to update the RZ domain from CPU to GPU. This does nothing.
    void update_device_RZ_domain() override {};

    // Wrapper method to perform MPI nonblocking send communication between CPUs
    // Return MPI error code
    inline int Isend(const real_t* buf, int count, MPI_Datatype datatype,
        int dest, int tag, MPI_Comm comm, MPI_Request* request) const override
    {
        return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    }

    // Wrapper method to perform MPI nonblocking receive communication
    // between CPUs. Return MPI error code.
    inline int Irecv(real_t* buf, int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm, MPI_Request* request) const override
    {
        return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    }

    // Wrapper method to perform blocking allreduce communication between CPUs
    // Return 0 for success and 1 for error
    inline int32_t Allreduce(const real_t* sendbuff, real_t* recvbuff,
                             size_t count, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm) const override
    {
        int32_t ierr = MPI_Allreduce(sendbuff, recvbuff, count,
                                     datatype, op, comm);

        return (int32_t) (ierr != MPI_SUCCESS);
    }
};

#endif
