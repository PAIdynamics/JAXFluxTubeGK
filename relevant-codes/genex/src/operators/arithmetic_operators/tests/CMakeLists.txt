add_pfunit_ctest(arithmetics-utests
    TEST_SOURCES test_arithmetic_operators.pf
    OTHER_SOURCES res/helpers_arithmetic_operators_m.f90
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
  add_pfunit_ctest(arithmetics-utests-gpu-cxx
      TEST_SOURCES test_arithmetic_operators.pf
      OTHER_SOURCES res/helpers_arithmetic_operators_m.f90
      MAX_PES 8
      LINK_LIBRARIES GENEXCore-Library
      LINK_LIBRARIES GENEXTestUtils-Library
  )
endif()

if(ENABLE_OPENACC)
  add_pfunit_ctest(arithmetics-utests-gpu-acc
      TEST_SOURCES test_arithmetic_operators.pf
      OTHER_SOURCES res/helpers_arithmetic_operators_m.f90
      MAX_PES 8
      LINK_LIBRARIES GENEXCore-Library
      LINK_LIBRARIES GENEXTestUtils-Library
  )
endif()

if(ENABLE_OPENMPX)
  add_pfunit_ctest(arithmetics-utests-gpu-ompx
      TEST_SOURCES test_arithmetic_operators.pf
      OTHER_SOURCES res/helpers_arithmetic_operators_m.f90
      MAX_PES 8
      LINK_LIBRARIES GENEXCore-Library
      LINK_LIBRARIES GENEXTestUtils-Library
  )
endif()

if(ENABLE_CUDA)
  add_pfunit_ctest(arithmetics-utests-gpu-cuda
      TEST_SOURCES test_arithmetic_operators.pf
      OTHER_SOURCES res/helpers_arithmetic_operators_m.f90
      MAX_PES 8
      LINK_LIBRARIES GENEXCore-Library
      LINK_LIBRARIES GENEXTestUtils-Library
  )
endif()
