target_sources(GENEXCore-CXX-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array_2d.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array_3d.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array_4d.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array_5d.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_array_dev_ptr.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_data_storage.cxx
        ${CMAKE_CURRENT_LIST_DIR}/helpers_mail_delivery.cxx
)

if(GENEX_CXX_ENABLE_GTESTS)
  add_executable(TestDataArray test_data_array.cxx)
  target_link_libraries(TestDataArray
    GENEXCore-CXX-Library
    GTest::gtest_main
    GTest::gmock_main
    )
  gtest_discover_tests(TestDataArray)
endif()
