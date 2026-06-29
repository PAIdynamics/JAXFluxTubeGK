#[=======================================================================[.rst:
AutogenerateTesting
-------------------

.. version:: 1.1

Autogenerate setup for MMS testing.

.. command:: autogenerate_testing

  .. code-block:: cmake

    autogenerate_testing()

  Autogenerate setup, parameter files, slurm job scripts for MMS testing.

#]=======================================================================]

function(autogenerate_testing)
  message(STATUS "${BoldBlue}Autogenerating unit testing resources "
                 "in ./unit-testing${ColourReset}")

  # Copy cicd/unit-testing folder to binary directory
  file(COPY ${CMAKE_SOURCE_DIR}/cicd/unit-testing
       DESTINATION ${CMAKE_BINARY_DIR})

  message(STATUS "${BoldBlue}Autogenerating MMS testing resources "
                 "in ./mms-testing${ColourReset}")

  # Fetch the supported runtime modes
  set(_runtime_modes "cpu")
  if(ENABLE_GPU)
    list(APPEND _runtime_modes "cxx")
  endif()
  if(ENABLE_OPENACC)
    list(APPEND _runtime_modes "acc")
  endif()
  if(ENABLE_OPENMPX)
    list(APPEND _runtime_modes "ompx")
  endif()

  # Copy python file to binary directory
  file(COPY ${CMAKE_SOURCE_DIR}/cicd/mms-testing/mms_assert.py
       DESTINATION ${CMAKE_BINARY_DIR}/mms-testing)
  file(COPY ${CMAKE_SOURCE_DIR}/cicd/mms-testing/mms_scaling_plot.py
       DESTINATION ${CMAKE_BINARY_DIR}/mms-testing)

  # Install python tools via pip
  execute_process(
    COMMAND python ${CMAKE_SOURCE_DIR}/cmake/python/generate_mms_testing.py
                   ${CMAKE_SOURCE_DIR}/cicd/mms-testing ${CMAKE_BINARY_DIR}
                   ${CMAKE_BUILD_TYPE}
                   ${_runtime_modes}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _outmsg
    ERROR_VARIABLE _errmsg
  )

  # Return warning message if errors occur
  if(_status AND NOT _status EQUAL 0)
    message(WARNING " autogenerate_testing:\n"
                    "   MMS-testing setup autogeneration failed!\n"
                    "   See automms.err for the error log.\n"
                    "   User may proceed building GENE-X.")

    file(WRITE ${CMAKE_BINARY_DIR}/automms.out "${_outmsg}")
    file(WRITE ${CMAKE_BINARY_DIR}/automms.err "${_errmsg}")
  endif()

  # Unset temporary variables
  unset(_runtime_mode)
  unset(_status)
  unset(_outmsg)
  unset(_errmsg)

endfunction()
