target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/math_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/quadrature_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/csralg_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/lagrange_interpolator_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/elliptic_integrals_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/chebyshev_polynomials_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/polynomial_integrals_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/bessel_m.f90
)

target_files(
    TARGET GENEXCore-Library
    FILES ${CMAKE_CURRENT_LIST_DIR}/fukushima_ceie.txt
          ${CMAKE_CURRENT_LIST_DIR}/fukushima_ceik.txt
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
