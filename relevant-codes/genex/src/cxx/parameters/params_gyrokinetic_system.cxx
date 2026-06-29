#include "params_gyrokinetic_system.hxx"

namespace params_gyrokinetic_system
{
    // Private objects with file scope and external linkage

    // Pointer to a flag activating the rhs of the gyrokinetic system
    static bool with_rhs {};

    // Pointer to a flag activating electromagnetic flutter in
    // the gyrokinetic system
    static bool with_em_flutter {};

    // Pointer to a flag activating nonlinear polarization in
    // the quasi-neutrality equation of the gyrokinetic system
    static bool with_nlin_polarization {};

    // Pointer to a flag activating the parallel magnetic field fluctuations
    // in the gyrokinetic system
    static bool with_bpar {};

    // Getters for the private objects

    bool get_with_rhs() { return with_rhs; }
    bool get_with_em_flutter() { return with_em_flutter; }
    bool get_with_nlin_polarization() { return with_nlin_polarization; }
    bool get_with_bpar() { return with_bpar; }
}

void cbind_set_params_gyrokinetic_system(const int32_t with_rhs,
                                         const int32_t with_em_flutter,
                                         const int32_t with_nlin_polarization,
                                         const int32_t with_bpar)
{
    params_gyrokinetic_system::with_rhs = static_cast<bool>(with_rhs);
    params_gyrokinetic_system::with_em_flutter =
        static_cast<bool>(with_em_flutter);
    params_gyrokinetic_system::with_nlin_polarization =
        static_cast<bool>(with_nlin_polarization);
    params_gyrokinetic_system::with_bpar =
        static_cast<bool>(with_bpar);
}
