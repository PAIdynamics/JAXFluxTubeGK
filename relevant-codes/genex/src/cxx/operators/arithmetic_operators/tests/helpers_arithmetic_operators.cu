#ifndef HELPERS_ARITHMETIC_OPERATORS_CU
#define HELPERS_ARITHMETIC_OPERATORS_CU

#include "genex_cxx_env.hxx"

#ifdef __cplusplus
extern "C" {
#endif

real_t cbind_ref_op_axpy_cuda(const real_t a, const real_t x,
                              const real_t y)
{
    return y + a * x;
}

real_t cbind_ref_op_lin_comb_cuda(const real_t a1, const real_t a2,
                                  const real_t x1, const real_t x2)
{
    return a1 * x1 + a2 * x2;
}

#ifdef __cplusplus
}
#endif

#endif
