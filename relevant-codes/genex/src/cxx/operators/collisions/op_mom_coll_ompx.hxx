#ifndef OP_MOM_COLL_OMPX_HXX
#define OP_MOM_COLL_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_mom_coll.hxx"
#include "data_array_ompx.hxx"
#include <mpi.h>
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_mom_coll_gpu_t with OpenMP offload on GPU
class op_mom_coll_ompx_t: public op_mom_coll_gpu_t
{
public:
    // Constructor of the OpenMP offload child class
    op_mom_coll_ompx_t(const dcomm_handler_t& dcomm_handler,
                       const mesh_5d_t& mesh)
    : op_mom_coll_gpu_t{dcomm_handler, mesh}
    {
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data \
            map(to: this[:1], \
                    this->size_mu, \
                    this->size_sp, \
                    this->PI, \
                    this->dens_floor, \
                    this->temp_floor, \
                    this->prefac_bps_ptr[:this->size_sp])

        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // Destructor of the OpenMP offload child class
    ~op_mom_coll_ompx_t() override
    {
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data \
            map(delete: this->prefac_bps_ptr[:this->size_sp], \
                        this->PI, \
                        this->temp_floor, \
                        this->dens_floor, \
                        this->size_sp, \
                        this->size_mu, \
                        this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Copy constructor is disabled
    op_mom_coll_ompx_t(const op_mom_coll_ompx_t&) = delete;

    // Copy-assignment operator is disabled
    op_mom_coll_ompx_t& operator=(const op_mom_coll_ompx_t&) = delete;

    int32_t apply(const dcomm_handler_t& dcomm_handler,
                  const mesh_5d_t& mesh,
                  const data_array_t<const real_t, 5>& f_in,
                  data_array_t<real_t, 4>& moments) const override
    {
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        data_array_ompx_t<real_t, 4> send_buffer(
            {lb_stripped[0], lb_stripped[1], 1, 1},
            {ub_stripped[0], ub_stripped[1], 3, mesh.get_size_sp()}, 0.0);

        profiler::start_region("reduction");

        #pragma omp target teams distribute parallel for simd collapse(3) \
            default(none) defaultmap(none) \
            shared(lb_stripped, ub_stripped, mesh, f_in, moments, send_buffer)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
            for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
            {
                this->comp_kernel_prior(i, k, l, m, n, mesh, f_in, send_buffer);
            }
        }

        profiler::end_region("reduction");

        profiler::start_allreduce_region(dcomm_handler.get_comm_vp_mu_sp());

        int32_t ierr =
            dcomm_handler.Allreduce(send_buffer.get_array_ptr(),
                                    moments.get_array_ptr(),
                                    send_buffer.get_size(),
                                    MPI_REAL_T, MPI_SUM,
                                    dcomm_handler.get_comm_vp_mu_sp());

        profiler::end_allreduce_region();

        profiler::start_region("store");

        // 2nd loop: apply corrections
        #pragma omp target teams distribute parallel for simd collapse(3) \
            default(none) defaultmap(none) \
            shared(lb_stripped, ub_stripped, moments)
        for (int32_t n = 1; n <= this->size_sp; n++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel_post(i, k, n, moments);
        }

        profiler::end_region("store");

        return ierr;
    }
};

#endif
