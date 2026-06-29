target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/helpers_dcomm_handler.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_dcomm_handler_domain.cxx
)

if(GENEX_CXX_ENABLE_GTESTS)
  add_executable(DCommHandler-utests test_dcomm_handler.cxx)
  target_link_libraries(DCommHandler-utests
    GENEXCore-CXX-Library
    GTest::gtest_main
    GTest::gmock_main
  )
  gtest_discover_tests(DCommHandler-utests)
  if (ENABLE_OPENACC)
    add_executable(DCommHandler-utests-gpu-acc test_dcomm_handler.cxx)
    target_compile_definitions(DCommHandler-utests-gpu-acc PRIVATE "ENABLE_OPENACC")
    target_link_libraries(DCommHandler-utests-gpu-acc
      GENEXCore-CXX-Library
      GTest::gtest_main
      GTest::gmock_main
    )
    gtest_discover_tests(DCommHandler-utests-gpu-acc)
  endif()
  if (ENABLE_OPENMPX)
    add_executable(DCommHandler-utests-gpu-ompx test_dcomm_handler.cxx)
    target_compile_definitions(DCommHandler-utests-gpu-ompx PRIVATE "ENABLE_OPENMPX")
    target_link_libraries(DCommHandler-utests-gpu-ompx
      GENEXCore-CXX-Library
      GTest::gtest_main
      GTest::gmock_main
    )
    gtest_discover_tests(DCommHandler-utests-gpu-ompx)
  endif()
endif()
