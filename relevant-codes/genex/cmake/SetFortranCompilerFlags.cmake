macro(set_fortran_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  set(Fortran_FLAGS)
  set(Fortran_FLAGS_DEBUG)
  set(Fortran_FLAGS_RELEASE)
  set(Linker_FLAGS)

  if(CMAKE_Fortran_COMPILER_ID MATCHES IntelLLVM OR
     CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    # Predefined compiler macro __INTEL_COMPILER
    set_fortran_intel_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    # Predefined compiler macro __GFORTRAN__
    set_fortran_gnu_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
    list(APPEND GENEX_COMPILE_DEFINITIONS "__GNU__")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES PGI)
    # Predefined compiler macro __PGI
    set_fortran_nvhpc_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES LLVMFlang)
    message(STATUS "Found LLVM/flang")
    set_fortran_llvm_compiler_flags(
      ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
      ${COVERAGE})
  endif()

  # [ConfiX] Register the options to ConfiX
  register_compiler_options(
    LANG Fortran
    COMMON_OPTIONS "Fortran_FLAGS"
    DEBUG_OPTIONS "Fortran_FLAGS_DEBUG"
    RELEASE_OPTIONS "Fortran_FLAGS_RELEASE"
    LINKER_OPTIONS "Linker_FLAGS")

  # [ConfiX] Sanitize the Fortran compiler options (only if -DSANITIZE_FLAGS=ON)
  sanitize_compiler_options(LANG Fortran DEFAULT OFF)

  # [ConfiX] Print default CMake Fortran compiler flags
  print_default_compiler_options(Fortran)

  # [ConfiX] Print GENE-X custom linker flags
  print_compiler_options(Fortran Linker)
endmacro()

macro(set_fortran_nvhpc_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  list(APPEND Linker_FLAGS
    -mp
  )

  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS "22.0")
    message(STATUS
      "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}"
      " doesn't support submodules")
    message(FATAL_ERROR
      "${BoldRed}Insufficient nvhpc version (minimum 22.0)${ColourReset}")
  endif()

  list(APPEND Fortran_FLAGS
    -Kieee
    -Mi4
    -Mdefaultunit
    -Mallocatable=03
    -Mpreprocess
    -mp
    -Mextend
    )

  list(APPEND Fortran_FLAGS_DEBUG
    -traceback
    -Mchkptr
    -Mchkstk
    -Mbounds
    )

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND Fortran_FLAGS_RELEASE ${NVHPC_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
      message(WARNING "No production run options set for nvhpc. "
                      "Ignoring -DPRODUCTION_RUN")
  endif()

  if(${COMPILER_REPORTS})
      message(WARNING "No compiler reports options set for nvhpc. "
                      "Ignoring -DCOMPILER_REPORTS")
  endif()

  if(${COVERAGE})
      message(WARNING "No coverage options set for nvhpc. Ignoring -DCOVERAGE")
  endif()

endmacro()


macro(set_fortran_intel_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  list(APPEND Fortran_FLAGS
    -fpp
    -g
    -no-wrap-margin
    )

  # Default debug flags valid for ifort and ifx
  list(APPEND Fortran_FLAGS_DEBUG
    -traceback
    -fstack-security-check
    -fpe0
    -O0
    "SHELL:-diag-disable 10182"
    "SHELL:-warn all"
    "SHELL:-check all,noarg_temp_created"
    "SHELL:-debug inline-debug-info"
    )

  # Set Linker_FLAGS_DEBUG only for debug mode since currently no
  # differentiation between linker flags is supported
  string(TOLOWER ${CMAKE_BUILD_TYPE} _btype)
  if(_btype STREQUAL "debug")
    list(APPEND Linker_FLAGS "SHELL:-check all")
  endif()

  # Some flags are treated differently for the ifx and ifort compilers
  if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "IntelLLVM")
    list(APPEND Fortran_FLAGS_DEBUG
      -ftrapv
      -Rno-debug-disables-optimization
      )
  else()
    list(APPEND Fortran_FLAGS_DEBUG
      -ftrapuv
      )
  endif()

  list(APPEND Fortran_FLAGS_RELEASE -O${OPTLEVEL})

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND Fortran_FLAGS_RELEASE ${Fortran_Intel_CPU_OPT_FLAGS})
  list(APPEND Fortran_FLAGS_RELEASE ${Fortran_Intel_ALIGN_FLAG})

  if(${PRODUCTION_RUN})
    list(APPEND Fortran_FLAGS_RELEASE -ipo)
    list(APPEND Linker_FLAGS -ipo)
  endif()

  if(${COMPILER_REPORTS})
    list(APPEND Fortran_FLAGS
      -qopt-report=3
      -qopt-report-phase=all
      -qopt-report-file=stdout
      )
  endif()

  if(${COVERAGE})
    message(STATUS "Adding flags for coverage to Fortran_FLAGS_DEBUG")
    list(APPEND Fortran_FLAGS_DEBUG -prof-gen=srcpos -prof-src-root=${CMAKE_SOURCE_DIR})
  endif()

endmacro()

macro(set_fortran_gnu_compiler_flags
    PRECISION OPTLEVEL CHIP PRODUCTION_RUN COMPILER_REPORTS COVERAGE)

  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS "6.0.0")
    message(STATUS
      "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}"
      " doesn't support submodules")
    message(FATAL_ERROR
      "${BoldRed}Insufficient gfortan version (minimum 6)${ColourReset}")
  endif()

  list(APPEND Fortran_FLAGS
    -fall-intrinsics
    -g
    -cpp
    -ffree-line-length-none
    )

  if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
    list(APPEND Fortran_FLAGS
      -fallow-argument-mismatch
      )
  endif()

  # NOTE: The recursion check is removed due to a known bug when compiling
  #       with OpenMP
  list(APPEND Fortran_FLAGS_DEBUG
    -fbacktrace
    -Wno-unused
    -Wno-surprising
    -Wno-unused-dummy-argument
    -ffpe-trap=invalid,zero,overflow
    -fcheck=all,no-recursion
    -Wall
    -Wno-array-temporaries)

  list(APPEND Fortran_FLAGS_RELEASE -O${OPTLEVEL})

  # Use CPU optimization flags defined by configure_cpu() from ConfiX
  list(APPEND Fortran_FLAGS_RELEASE ${GNU_CPU_OPT_FLAGS})

  if(${PRODUCTION_RUN})
    list(APPEND Fortran_FLAGS_RELEASE -fipa-pta)
  endif()

  if(${COMPILER_REPORTS})
    list(APPEND Fortran_FLAGS
      -fopt-info-all-optall=genex.optrpt
      )
  endif()

  if(${COVERAGE})
      # NOTE: --coverage is synonymous to -fprofile-arcs -ftest-coverage when
      #       compiling and -lgcov when linking.
      list(APPEND Fortran_FLAGS_DEBUG --coverage -fprofile-abs-path)
      list(APPEND Linker_FLAGS --coverage)
  endif()

endmacro()

macro(set_fortran_llvm_compiler_flags)

  list(APPEND Fortran_FLAGS
    -g
    -cpp
  )

  list(APPEND Fortran_FLAGS_DEBUG -Wall)

  list(APPEND Fortran_FLAGS_RELEASE -O${OPTLEVEL})
endmacro()
