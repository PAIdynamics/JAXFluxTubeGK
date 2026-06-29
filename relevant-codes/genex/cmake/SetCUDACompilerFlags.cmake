macro(set_cuda_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)
  # Set CUDA_FLAGS, CUDA_FLAGS_DEBUG, CUDA_FLAGS_RELEASE

  if(CMAKE_CUDA_COMPILER_ID MATCHES NVIDIA)
    set_cuda_nvhpc_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  endif()

  # [ConfiX] Register the options to ConfiX
  register_compiler_options(
    LANG CUDA
    COMMON_OPTIONS "CUDA_FLAGS"
    DEBUG_OPTIONS "CUDA_FLAGS_DEBUG"
    RELEASE_OPTIONS "CUDA_FLAGS_RELEASE")

  # [ConfiX] Print default CMake compiler flags
  print_default_compiler_options(CUDA)

  # [ConfiX] Print GENE-X custom compiler flags
  print_compiler_options(CUDA)
endmacro()

macro(set_cuda_nvhpc_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(CUDA_FLAGS)
  set(CUDA_FLAGS_DEBUG)
  set(CUDA_FLAGS_RELEASE)

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR
      "${BoldRed}Insufficient NVCC version (minimum 11.0)${ColourReset}")
  endif()

  list(APPEND CUDA_FLAGS
    -g
    --fmad=false
    )

  list(APPEND CUDA_FLAGS_DEBUG
    "SHELL:-Wall"
    )

  if(${PRODUCTION_RUN})
      message(WARNING "No production run options set for NVHPCSDK. "
                      "Ignoring -DPRODUCTION_RUN")
  endif()

  if(${COMPILER_REPORTS})
      message(WARNING "No compiler reports options set for NVHPCSDK. "
                      "Ignoring -DCOMPILER_REPORTS")
  endif()

  if(${COVERAGE})
      message(WARNING "No coverage options set for NVHPCSDK. "
                      "Ignoring -DCOVERAGE")
  endif()

endmacro()
