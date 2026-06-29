#ifndef PARAMS_GYROKINECTIC_SYSTEM_HXX
#define PARAMS_GYROKINECTIC_SYSTEM_HXX

#include "genex_cxx_env.hxx"

// Namespace for parameters related to the gyrokinetic system
namespace params_gyrokinetic_system
{
    // Getter for a flag to activate the rhs of the gyrokinetic system
    bool get_with_rhs();

    // Getter for a flag to activate electromagnetic flutter in
    // the gyrokinetic system
    bool get_with_em_flutter();

    // Getter for a flag to activate nonlinear polarization in
    // the quasi-neutrality equation of the gyrokinetic system
    bool get_with_nlin_polarization();

    // Getter for a flag to activate the parallel magnetic field fluctuations
    // in the gyrokinetic system
    bool get_with_bpar();
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_set_params_gyrokinetic_system(const int32_t with_rhs,
                                         const int32_t with_em_flutter,
                                         const int32_t with_nlin_polarization,
                                         const int32_t with_bpar);

#ifdef __cplusplus
}
#endif

#endif
