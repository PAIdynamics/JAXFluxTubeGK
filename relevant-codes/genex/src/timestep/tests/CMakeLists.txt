# NOTE: The name of this test must be not "timestep-utests" and it must be
#       different from the file name. Otherwise this results in compile problems
#       with ifx compiler and debug mode on the Raven machine.

set(HELPER_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/res/helpers_timestep_m.f90
)

add_pfunit_ctest(timestepping-utests
    TEST_SOURCES test_timestep.pf
    OTHER_SOURCES ${HELPER_SOURCES}
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(timestepping-utests-gpu-cxx
        TEST_SOURCES test_timestep.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(timestepping-utests-gpu-acc
        TEST_SOURCES test_timestep.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(timestepping-utests-gpu-ompx
        TEST_SOURCES test_timestep.pf
        OTHER_SOURCES ${HELPER_SOURCES}
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
