add_pfunit_ctest(mesh-utests
    TEST_SOURCES test_mesh_5d.pf test_structured_grids.pf
    MAX_PES 8
    LINK_LIBRARIES GENEXCore-Library
    LINK_LIBRARIES GENEXTestUtils-Library
)

if(ENABLE_GPU)
    add_pfunit_ctest(mesh-utests-gpu-cxx
        TEST_SOURCES test_mesh_5d_gpu.pf
        OTHER_SOURCES res/helpers_mesh_5d_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENACC)
    add_pfunit_ctest(mesh-utests-gpu-acc
        TEST_SOURCES test_mesh_5d_gpu.pf
        OTHER_SOURCES res/helpers_mesh_5d_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()

if(ENABLE_OPENMPX)
    add_pfunit_ctest(mesh-utests-gpu-ompx
        TEST_SOURCES test_mesh_5d_gpu.pf
        OTHER_SOURCES res/helpers_mesh_5d_gpu_m.f90
        MAX_PES 8
        LINK_LIBRARIES GENEXCore-Library
        LINK_LIBRARIES GENEXTestUtils-Library
    )
endif()
