#ifndef PARAMS_COLLISION_HXX
#define PARAMS_COLLISION_HXX

#include "genex_cxx_env.hxx"
#include <string>

// Namespace for parameters related to the collision models
namespace params_collisions
{
    // NOTE: Rollback to std::string from std::string_view (C++17) because
    //       the classic Intel C++ compiler (icpc) does not support
    //       std::string_view. Implement back std::string_view when GENE-X
    //       supports Intel LLVM compiler (icpx).

    // Getter for the type of the collision operator
    std::string& get_coll_type();

    // Getter for the relaxation type
    std::string& get_relx_type();

    // Getter for the floor of density
    real_t get_dens_floor();

    // Getter for the floor of temperature
    real_t get_temp_floor();
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_set_params_collisions(char* coll_type, char* relx_type,
                                 real_t dens_floor, real_t temp_floor);

#ifdef __cplusplus
}
#endif

#endif
