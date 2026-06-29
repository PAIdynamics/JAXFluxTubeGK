target_sources(GENEXCore-Library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/part_handler_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/diag_files_m.f90
        ${CMAKE_CURRENT_LIST_DIR}/checkpoint_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/checkpoint_neut_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/em_fields_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mom_0d_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mom_2d_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/neutrals_file_s.f90
        ${CMAKE_CURRENT_LIST_DIR}/mesh_file_s.f90
)

if(ENABLE_UTESTS)
  add_subdirectory(tests)
endif()
