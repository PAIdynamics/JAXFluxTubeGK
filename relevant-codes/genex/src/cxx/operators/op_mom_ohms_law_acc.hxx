#ifndef OP_MOM_OHMS_LAW_ACC_HXX
#define OP_MOM_OHMS_LAW_ACC_HXX

#include "genex_cxx_env.hxx"
#include "op_mom_ohms_law.hxx"
#include "data_array_acc.hxx"
#include <mpi.h>
#include <openacc.h>

// C++ class which corresponds to the Fortran class op_mom_ohms_law_gpu_t
// with OpenACC
class op_mom_ohms_law_acc_t: public op_mom_ohms_law_gpu_t
{
public:
    // Constructor of the OpenACC child class
    op_mom_ohms_law_acc_t(const mesh_5d_t& mesh)
    : op_mom_ohms_law_gpu_t{mesh}
    {
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma acc enter data \
            copyin(this[:1], \
                   this->size_sp, \
                   this->PI, \
                   this->prefac_cd_vp, \
                   this->prefac_bps_ptr[:this->size_sp], \
                   this->prefac_norm_lambda_ptr[:this->size_sp], \
                   this->prefac_norm_b_ptr[:this->size_sp])

        this->dmem_debug(dmd::mode_t::ALLOC);
    };

    // Destructor of the OpenACC child class
    ~op_mom_ohms_law_acc_t() override
    {
        // Deallocate the device pointers of the class members from the device
        #pragma acc exit data \
            delete(this->prefac_norm_b_ptr[:this->size_sp], \
                   this->prefac_norm_lambda_ptr[:this->size_sp], \
                   this->prefac_bps_ptr[:this->size_sp], \
                   this->prefac_cd_vp, \
                   this->PI, \
                   this->size_sp, \
                   this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    };

    // Copy constructor is disabled
    op_mom_ohms_law_acc_t(const op_mom_ohms_law_acc_t&) = delete;

    // Copy-assignment operator is disabled
    op_mom_ohms_law_acc_t& operator=(const op_mom_ohms_law_acc_t&) = delete;

    int32_t apply(
        const mesh_5d_t& mesh,
        const dcomm_handler_t& dcomm_handler,
        const data_array_t<const real_t, 5>& f_in,
        const data_array_t<const real_t, 5>& dfdt_in,
        data_array_t<real_t, 2>& lambda_ohms_law,
        data_array_t<real_t, 2>& b_ohms_law) const override
    {
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        data_array_acc_t<real_t, 3> send_buffer(
            {lb_stripped[0], lb_stripped[1], 1},
            {ub_stripped[0], ub_stripped[1], 2}, 0.0);
        data_array_acc_t<real_t, 3> receive_buffer(
            {lb_stripped[0], lb_stripped[1], 1},
            {ub_stripped[0], ub_stripped[1], 2});

        profiler::start_region("reduction");

        #pragma acc parallel default(none) \
            present(lb_stripped, ub_stripped, \
                    this, mesh, f_in, dfdt_in, send_buffer)
        // NOTE: send_buffer reduction prevents parallelization on vp, mu,
        //       species direction.
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        #pragma acc loop independent gang vector collapse(2)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel_prior(i, k, l, m, n,
                                    mesh, f_in, dfdt_in, send_buffer);
        }

        profiler::end_region("reduction");

        profiler::start_allreduce_region(dcomm_handler.get_comm_vp_mu_sp());

        int32_t ierr =
            dcomm_handler.Allreduce(send_buffer.get_array_ptr(),
                                    receive_buffer.get_array_ptr(),
                                    send_buffer.get_size(),
                                    MPI_REAL_T, MPI_SUM,
                                    dcomm_handler.get_comm_vp_mu_sp());

        profiler::end_allreduce_region();

        profiler::start_region("store");

        // Copy information from MPI buffers into the output arrays
        #pragma acc parallel default(none) \
            present(lb_stripped, ub_stripped, \
                    this, mesh, lambda_ohms_law, b_ohms_law, receive_buffer)
        #pragma acc loop independent gang vector collapse(2)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel_post(i, k,
                mesh, receive_buffer, lambda_ohms_law, b_ohms_law);
        }

        profiler::end_region("store");

        return ierr;
    }
};

#endif
