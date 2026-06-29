macro(set_cxx_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)
  # Set CXX_FLAGS, CXX_FLAGS_DEBUG, CXX_FLAGS_RELEASE

  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    # Predefined compiler macro __INTEL_COMPILER
    set_cxx_intel_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    # Predefined compiler macro __GNU__
    set_cxx_gnu_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
    list(APPEND GENEX_COMPILE_DEFINITIONS "__GNU__")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    set_cxx_nvhpc_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES PGI)
    # Predefined compiler macro __PGI
    set_cxx_pgi_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set_cxx_llvm_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  endif()

  # [ConfiX] Register the options to ConfiX
  register_compiler_options(
    LANG CXX
    COMMON_OPTIONS "CXX_FLAGS"
    DEBUG_OPTIONS "CXX_FLAGS_DEBUG"
    RELEASE_OPTIONS "CXX_FLAGS_RELEASE"
    LINKER_OPTIONS "Linker_FLAGS")

  # [ConfiX] Sanitize the CXX compiler options (only if -DSANITIZE_FLAGS=ON)
  sanitize_compiler_options(LANG CXX DEFAULT OFF)

  # [ConfiX] Print default CMake CXX compiler flags
  print_default_compiler_options(CXX)

  # [ConfiX] Print GENE-X custom CXX compiler flags
  print_compiler_options(CXX)

  # [ConfiX] Print GENE-X custom linker flags
  print_compiler_options(Fortran Linker)
endmacro()

macro(set_cxx_llvm_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CXX_FLAGS)
  set(CXX_FLAGS_DEBUG)
  set(CXX_FLAGS_RELEASE)

  list(APPEND CXX_FLAGS
    -g
    )

  list(APPEND CXX_FLAGS_DEBUG
    -Wall
    -ffp-contract=off
    )

  list(APPEND CXX_FLAGS_RELEASE
    -ffp-contract=fast
    -finline-functions
    -finline-hint-functions
    -O${OPTLEVEL}
    )

  # OpenMP offload specific compiler and linker options
  if(ENABLE_OPENMPX)
    if(GPU_VENDOR MATCHES AMD)
      list(APPEND CXX_FLAGS -fopenmp --offload-arch=${LLVM_TARGET_ARCH})
    elseif(GPU_VENDOR MATCHES NVIDIA)
      list(APPEND CXX_FLAGS -Xopenmp-target -march=${LLVM_TARGET_ARCH}
                            -fopenmp -fopenmp-targets=${LLVM_TARGET_TRIPLES}
                            --cuda-path=$ENV{CUDA_HOME})
    else()
      message(FATAL_ERROR "OpenMP offload is not supported on "
                          "\"${GPU_VENDOR}\" GPU!")
    endif()
  endif()

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND CXX_FLAGS_RELEASE ${LLVM_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
      message(WARNING "No production run options set for LLVM. "
                      "Ignoring -DPRODUCTION_RUN")
  endif()

  if(${COMPILER_REPORTS})
      message(WARNING "No compiler reports options set for LLVM. "
                      "Ignoring -DCOMPILER_REPORTS")
  endif()

  if(COVERAGE)
      message(WARNING "No coverage options set for LLVM. Ignoring -DCOVERAGE")
  endif()

endmacro()

macro(set_cxx_nvhpc_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CXX_FLAGS)
  set(CXX_FLAGS_DEBUG)
  set(CXX_FLAGS_RELEASE)

  set(_cuda_ver_docstr "CUDA version used from NVHPC SDK")

  # Set default NVHPC_CUDA_VERSION if GENEX_CUDA_VERSION environment variable
  # is defined
  if(DEFINED ENV{GENEX_CUDA_VERSION})
    set(NVHPC_CUDA_VERSION $ENV{GENEX_CUDA_VERSION}
        CACHE STRING ${_cuda_ver_docstr})
  else()
    set(NVHPC_CUDA_VERSION CACHE STRING ${_cuda_ver_docstr})
  endif()

  # Unset NVHPC_CUDA_VERSION if defined and set as "default"
  if(DEFINED NVHPC_CUDA_VERSION)
    if(NVHPC_CUDA_VERSION MATCHES "default")
      unset(NVHPC_CUDA_VERSION CACHE)
    endif()
  endif()

  # Help CMake to find the path to NVHPCConfig.cmake
  if(NOT DEFINED NVHPC_DIR OR NOT NVHPC_DIR MATCHES "$ENV{NVHPC_HOME}")
    if(DEFINED ENV{NVHPC_ROOT})
      set(NVHPC_DIR "$ENV{NVHPC_ROOT}/cmake")
    else()
      string(REPLACE "/compilers/" ";" _nvhpc_list ${CMAKE_CXX_COMPILER})
      list(GET _nvhpc_list 0 NVHPC_DIR)
      set(NVHPC_DIR "${NVHPC_DIR}/cmake")
      unset(_nvhpc_list)
    endif()
  endif()

  # Find useful toolkits in NVIDIA HPC SDK
  find_package(NVHPC REQUIRED COMPONENTS CUDA;MATH;HOSTUTILS;NCCL;PROFILER)
  get_target_property(NVHPC_CUDA_INCLUDE_DIR NVHPC::CUDA INTERFACE_INCLUDE_DIRECTORIES)

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0")
    message(STATUS
      "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
      " doesn't support submodules")
    message(FATAL_ERROR
      "${BoldRed}Insufficient NVHPCSDK version (minimum 19.0)${ColourReset}")
  endif()

  list(APPEND CXX_FLAGS
    -g
    -Kieee
    -Mpreprocess
    -mcmodel=medium
    )

  list(APPEND CXX_FLAGS_DEBUG
    -traceback
    "SHELL:-Wall"
    -Minfo=all
    -Mchkstk
    -Mnofma
    )

  list(APPEND CXX_FLAGS_RELEASE
    -fast
    -Mautoinline
    -O${OPTLEVEL}
    )

  list(APPEND Linker_FLAGS -mcmodel=medium)

  # OpenACC specific compiler and linker options
  if(ENABLE_OPENACC)
    list(APPEND CXX_FLAGS -acc=gpu)
  endif()

  # OpenMP offload specific compiler and linker options
  if(ENABLE_OPENMPX)
    list(APPEND CXX_FLAGS -mp=gpu)
  endif()

  # Common compiler and linker options for OpenACC and OpenMP offload
  if(ENABLE_OPENACC OR ENABLE_OPENMPX)
    list(APPEND CXX_FLAGS -target=gpu -gpu=${NVHPC_TARGET_ARCH})
  endif()

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND CXX_FLAGS_RELEASE ${NVHPC_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
      message(WARNING "No production run options set for NVHPCSDK. "
                      "Ignoring -DPRODUCTION_RUN")
  endif()

  if(${COMPILER_REPORTS})
    list(APPEND CXX_FLAGS -Minfo=accel,loop,mp)
  endif()

  if(COVERAGE)
      message(WARNING "No coverage options set for NVHPCSDK. "
                      "Ignoring -DCOVERAGE")
  endif()

endmacro()

macro(set_cxx_pgi_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CXX_FLAGS)
  set(CXX_FLAGS_DEBUG)
  set(CXX_FLAGS_RELEASE)

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0")
    message(STATUS
      "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
      " doesn't support submodules")
    message(FATAL_ERROR
      "${BoldRed}Insufficient pgf90 version (minimum 19.0)${ColourReset}")
  endif()

  list(APPEND CXX_FLAGS
    -g
    -Kieee
    -Mpreprocess
    -mcmodel=medium
    )

  list(APPEND CXX_FLAGS_DEBUG
    -traceback
    "SHELL:-Wall"
    -Minfo=all
    -Mchkstk
    -Mnofma
    )

    list(APPEND CXX_FLAGS_RELEASE
    -fast
    -Mautoinline
    -O${OPTLEVEL}
    )

  list(APPEND Linker_FLAGS -mcmodel=medium)

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND CXX_FLAGS_RELEASE ${NVHPC_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
      message(WARNING "No production run options set for pgi. "
                      "Ignoring -DPRODUCTION_RUN")
  endif()

  if(${COMPILER_REPORTS})
      message(WARNING "No compiler reports options set for pgi."
                      "Ignoring -DCOMPILER_REPORTS")
  endif()

  if(COVERAGE)
      message(WARNING "No coverage options set for pgi. Ignoring -DCOVERAGE")
  endif()

endmacro()


macro(set_cxx_intel_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CXX_FLAGS)
  set(CXX_FLAGS_DEBUG)
  set(CXX_FLAGS_RELEASE)

  # disable diag 161 "unrecognized pragma"
  # disable diag 10441 "deprecation warning for icpc"
  list(APPEND CXX_FLAGS
    -g
    "SHELL:-diag-disable 7416,7025,10121,161,10441"
    )

  list(APPEND CXX_FLAGS_DEBUG
    -traceback
    -O0
    "SHELL:-Wall"
    "SHELL:-Wno-unknown-pragmas"
    -fstack-security-check
    "SHELL:-debug inline-debug-info")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
    list(APPEND CXX_FLAGS_DEBUG
      -static-libgcc
      -Rno-debug-disables-optimization
      )
  else()
    list(APPEND CXX_FLAGS_DEBUG
      "SHELL:-diag-disable 10182"
      -ftrapuv
      )
  endif()

  list(APPEND CXX_FLAGS_RELEASE -O${OPTLEVEL})

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND CXX_FLAGS_RELEASE ${CXX_Intel_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
    list(APPEND CXX_FLAGS_RELEASE -ip)
  endif()

  if(${COMPILER_REPORTS})
    list(APPEND CXX_FLAGS
      -qopt-report=5
      -qopt-report-phase=all
      -qopt-report-per-object
      )
  endif()

  if(COVERAGE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
      list(APPEND CXX_FLAGS -fprofile-instr-generate)
    else()
      list(APPEND CXX_FLAGS -prof-gen=srcpos -prof-src-root=${CMAKE_SOURCE_DIR})
    endif()
  endif()

endmacro()

macro(set_cxx_gnu_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CXX_FLAGS)
  set(CXX_FLAGS_DEBUG)
  set(CXX_FLAGS_RELEASE)

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0.0")
    message(STATUS
      "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
      " doesn't support submodules")
    message(FATAL_ERROR
      "${BoldRed}Insufficient gfortan version (minimum 6)${ColourReset}")
  endif()

  list(APPEND CXX_FLAGS
    -g
    -cpp)

  list(APPEND CXX_FLAGS_DEBUG
    -Wno-unused
    -Wall)

  list(APPEND CXX_FLAGS_RELEASE -O${OPTLEVEL})

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND CXX_FLAGS_RELEASE ${GNU_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
    list(APPEND CXX_FLAGS_RELEASE -fipa-pta)
  endif()

  if(${COMPILER_REPORTS})
    list(APPEND CXX_FLAGS
      -fopt-info-all-optall=gene3d.optrpt
      )
  endif()

  if(COVERAGE)
    list(APPEND CXX_FLAGS --coverage)
  endif()

endmacro()
