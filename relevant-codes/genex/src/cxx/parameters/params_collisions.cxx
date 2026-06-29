#include "params_collisions.hxx"

namespace params_collisions
{
    // Private objects with file scope and external linkage

    // Collision operator to be used. Current options are: "none", "bgk",
    // "lbd" and "fpl"
    static std::string coll_type = "none";

    // Relaxation type used for the collision frequency. Options are either
    // "et" (entropic, temperature relaxation) or "em" (entropic, momentum
    // relaxation)
    static std::string relx_type = "em";

    // Parameter that sets a minimum limit on the density in the calculations
    // of collisional moments. If the calculated density is below this value,
    // it is replaced by it. This is needed because density is used as a
    // normalization for the other collisional moments
    static real_t dens_floor = 0.0;

    // Parameter that sets a minimum limit on the temperature
    // in the calculations of collisional moments. This is needed to prevent
    // possible negative temperatures that produce floating point exceptions
    // in the sqrt.
    static real_t temp_floor = real_eps;

    // Getters for the private objects

    std::string& get_coll_type() { return coll_type; }
    std::string& get_relx_type() { return relx_type; }
    real_t get_dens_floor() { return dens_floor; }
    real_t get_temp_floor() { return temp_floor; }
}

void cbind_set_params_collisions(char* coll_type, char* relx_type,
                                 real_t dens_floor, real_t temp_floor)
{
    params_collisions::coll_type  = std::string{coll_type};
    params_collisions::relx_type  = std::string{relx_type};
    params_collisions::dens_floor = dens_floor;
    params_collisions::temp_floor = temp_floor;
}
