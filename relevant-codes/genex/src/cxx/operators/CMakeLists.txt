target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/op_rhs_vlasov_eq_dynamic.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_rhs_vlasov_eq_static.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_maxwells_eq.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_mom_ohms_law.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_bnd_cond_dir.cxx
        ${CMAKE_CURRENT_LIST_DIR}/op_bnd_cond_neum.cxx
)

# NOTE: The following interface is necessary for the helpers to find
#       the header file of the operators.
target_include_directories(GENEXCore-CXX-Library
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

add_subdirectory(arithmetic_operators)
add_subdirectory(collisions)
