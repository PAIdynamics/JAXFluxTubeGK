if(ENABLE_GPU)
    set(HELPER_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_collog_m.f90
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_collision_operators_m.f90
    )
else()
    set(HELPER_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/res/helpers_collision_operators_m.f90
    )
endif()

add_pfunit_ctest(collision-utests
    TEST_SOURCES test_mom_coll_operators.pf
                 test_collog.pf
                 test_collision_operators.pf
    OTHER_SOURCES ${HELPER_SOURCES}
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(collision-utests-gpu-cxx
        TEST_SOURCES test_mom_coll_operators.pf
                     test_collog.pf
                     test_collision_operators.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(collision-utests-gpu-acc
        TEST_SOURCES test_mom_coll_operators.pf
                     test_collog.pf
                     test_collision_operators.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(collision-utests-gpu-ompx
        TEST_SOURCES test_mom_coll_operators.pf
                     test_collog.pf
                     test_collision_operators.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
