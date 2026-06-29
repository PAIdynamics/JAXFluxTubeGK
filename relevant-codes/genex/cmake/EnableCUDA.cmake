#[=======================================================================[.rst:
EnableCUDA
--------------------

Configure the project setup to enable CUDA language support

.. command:: enable_language_cuda

  .. code-block:: cmake

    enable_language_CUDA()

  Enable CUDA language support in GENE-X and configure CMake variables
  :variable:`CMAKE_CUDA_COMPILER`, :variable:`CMAKE_CUDA_ARCHITECTURES`,
  :variable:`CMAKE_CUDA_STANDARD`.

Updated variables
^^^^^^^^^^^^^^^^

``CMAKE_CUDA_COMPILER``
    The full path to the CUDA compiler nvcc.

``CMAKE_CUDA_ARCHITECTURES``
    The global list of CUDA architectures to generate the device code for.

``CMAKE_CUDA_STANDARD``
    The global value for The CUDA/C++ standard whose features are requested to
    build targets.

#]=======================================================================]

macro(enable_language_CUDA)
  # Case when NVHPC SDK is used
  if(DEFINED ENV{NVHPC_HOME})
    # Set CUDA version cache variable
    set(NVHPC_CUDA_VERSION "default" CACHE STRING
        "The CUDA version to use from NVIDIA HPC SDK")

    # FInd CUDAToolkit and get the correct full path to CUDA compiler nvcc
    set(CUDAToolkit_ROOT "${NVHPC_CUDA_INCLUDE_DIR}/.." )
    if(NOT (NVHPC_CUDA_VERSION STREQUAL "default"))
      find_package(CUDAToolkit ${NVHPC_CUDA_VERSION} REQUIRED)
    else()
      find_package(CUDAToolkit REQUIRED)
    endif()
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  endif()

  # Check CUDA compiler availability
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "${BoldRed}CUDA compiler cannot be found!${ColourReset}")
  endif()

  # Enable CMake language support for CUDA
  enable_language(CUDA)

  # Set CUDA architecture
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_TARGET_ARCH})
  message(STATUS "${BoldGreen}CUDA architectures set to "
                 "${CMAKE_CUDA_ARCHITECTURES}${ColourReset}")

  # Set CUDA standard
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    if(DEFINED CMAKE_CXX_STANDARD)
      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    else()
      set(CMAKE_CUDA_STANDARD 11)
    endif()
  endif()
  message(STATUS "${BoldGreen}CUDA standard is set to ${CMAKE_CUDA_STANDARD}"
                 "${ColourReset}")

endmacro()
