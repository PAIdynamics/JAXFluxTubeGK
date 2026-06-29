#ifndef OP_RHS_VLASOV_EQ_STATIC_OMP_HXX
#define OP_RHS_VLASOV_EQ_STATIC_OMP_HXX

#include "genex_cxx_env.hxx"
#include "op_rhs_vlasov_eq_static.hxx"
#include "data_array_factory.hxx"
#include <omp.h>
#include <array>

// C++ class which corresponds to the Fortran class
// op_rhs_vlasov_eq_static_gpu_t with OpenMP on CPU
class op_rhs_vlasov_eq_static_omp_t: public op_rhs_vlasov_eq_static_gpu_t
{
public:
    // Constructor of the OpenMP child class
    op_rhs_vlasov_eq_static_omp_t(const mesh_5d_t& mesh)
    : op_rhs_vlasov_eq_static_gpu_t{mesh} {};

    // Destructor of the OpenMP child class
    ~op_rhs_vlasov_eq_static_omp_t() override {};

    // Copy constructor is disabled
    op_rhs_vlasov_eq_static_omp_t(const op_rhs_vlasov_eq_static_omp_t&)
        = delete;

    // Copy-assignment operator is disabled
    op_rhs_vlasov_eq_static_omp_t&
        operator=(const op_rhs_vlasov_eq_static_omp_t&) = delete;

    int32_t apply(const mesh_5d_t& mesh,
                  const data_array_t<const real_t, 5>& f_in,
                  const data_array_t<const real_t, 2>& phi_in,
                  const data_array_t<const real_t, 2>& A_par_in,
                  const data_array_t<const real_t, 2>& B_par_in,
                  data_array_t<real_t, 5>& f_out) const override
    {
        const int32_t (&lb)[2] = phi_in.get_lbound();
        const int32_t (&ub)[2] = phi_in.get_ubound();
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        // Instantiate local 2D array objects for second order Hamiltonian
        const std::array<int32_t, 2> lb_H2 {lb[0], lb[1]};
        const std::array<int32_t, 2> ub_H2 {ub[0], ub[1]};
        data_array_t<real_t, 2>& H2 =
            *(data_array::create<real_t, 2>(lb_H2, ub_H2));
        bool da_err = data_array::is_erroneous;

        #pragma omp parallel default(none) shared(lb, ub, mesh, phi_in, H2)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            this->initialize_H2(i, k, mesh, phi_in, H2);
        }

        #pragma omp parallel default(none) \
            shared(lb_stripped, ub_stripped, \
                   mesh, f_in, phi_in, A_par_in, B_par_in, H2, f_out)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma omp for schedule(static) nowait
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel(i, k, l, m, n,
                              mesh, f_in, phi_in, A_par_in, B_par_in, H2, \
                              f_out);
        }

        delete &H2;
        da_err = da_err || data_array::is_erroneous;
        return (int32_t) da_err;
    }
};

#endif
