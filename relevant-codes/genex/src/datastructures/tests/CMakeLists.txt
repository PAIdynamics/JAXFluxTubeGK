add_pfunit_ctest(datastructure-utests
    TEST_SOURCES test_data_array.pf
                 test_mail_delivery.pf
                 test_data_storage.pf
    OTHER_SOURCES res/helpers_mail_delivery.f90
                  res/helpers_data_storage.f90
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    target_sources(datastructure-utests
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_mail_delivery_gpu_m.f90
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_data_storage_gpu_m.f90
    )

    add_pfunit_ctest(datastructure-utests-gpu-cxx
        TEST_SOURCES test_data_array_gpu.pf
                     test_mail_delivery.pf
                     test_data_storage.pf
        OTHER_SOURCES res/helpers_data_array_gpu_m.f90
                      res/helpers_mail_delivery_gpu_m.f90
                      res/helpers_mail_delivery.f90
                      res/helpers_data_storage.f90
                      res/helpers_data_storage_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(datastructure-utests-gpu-acc
        TEST_SOURCES test_data_array_gpu.pf
                     test_mail_delivery.pf
                     test_data_storage.pf
        OTHER_SOURCES res/helpers_data_array_gpu_m.f90
                      res/helpers_mail_delivery_gpu_m.f90
                      res/helpers_mail_delivery.f90
                      res/helpers_data_storage.f90
                      res/helpers_data_storage_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(datastructure-utests-gpu-ompx
        TEST_SOURCES test_data_array_gpu.pf
                     test_mail_delivery.pf
                     test_data_storage.pf
        OTHER_SOURCES res/helpers_data_array_gpu_m.f90
                      res/helpers_mail_delivery_gpu_m.f90
                      res/helpers_mail_delivery.f90
                      res/helpers_data_storage.f90
                      res/helpers_data_storage_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
