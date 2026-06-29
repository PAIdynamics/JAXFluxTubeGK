target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/mailbox_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/mail_delivery_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/mail_delivery_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_array_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_array_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_cpu_2d_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_cpu_4d_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_cpu_5d_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/data_storage_5d_s.f90
)

if(ENABLE_GPU)
    target_sources(GENEXCore-Library
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/data_array_gpu_m.f90
            ${CMAKE_CURRENT_LIST_DIR}/data_storage_gpu_2d_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/data_storage_gpu_4d_s.f90
            ${CMAKE_CURRENT_LIST_DIR}/data_storage_gpu_5d_s.f90
    )
endif()

if(ENABLE_UTESTS)
    add_subdirectory(tests)
endif()
