#[=======================================================================[.rst:
InstallGENEXPrebuilderTools
---------------------------

.. version:: 1.0

Install GENE-X Python prebuilder tools via pip.

.. command:: install_genex_prebuilder_tools

  .. code-block:: cmake

    install_genex_prebuilder_tools()

  Install GENE-X Python prebuilder tools via pip.

#]=======================================================================]

function(install_genex_prebuilder_tools PREBUILDER_REQUIRED)
  message(STATUS "${BoldBlue}Installing GENE-X prebuilder python tools based "
                 "on ConfiX python tools${ColourReset}")

  # [ConfiX] Install ConfiX python tools
  install_confix_python_tools()

  # Copy files to binary directory
  file(COPY ${CMAKE_SOURCE_DIR}/cmake/python/
       DESTINATION ${CMAKE_BINARY_DIR}/lib_python/genex_prebuilder)

  # Install python tools via pip
  execute_process(
    COMMAND pip install -e ${CMAKE_BINARY_DIR}/lib_python/genex_prebuilder
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _outmsg
    ERROR_VARIABLE _errmsg
  )

  if(_status AND NOT _status EQUAL 0)
    file(WRITE ${CMAKE_BINARY_DIR}/prebuilder_pyinstall.err "${_errmsg}")
    if(PREBUILDER_REQUIRED)
      # Print installed python packages to help debug issues
      execute_process(
          COMMAND pip list
          OUTPUT_VARIABLE _piplist
          ERROR_QUIET
        )
      message(STATUS
              "${BoldCyan} Installed python packages: ${ColourReset}\n"
              "   ${_piplist}")

      # If confix installation failed and prebuilder required, we print the
      # error message to terminal forconvenience
      if(DEFINED _CONFIX_PYERR)
        execute_process(
            COMMAND cat ${CMAKE_BINARY_DIR}/confix_pyinstall.err
            OUTPUT_VARIABLE _confixmsg
            ERROR_QUIET
          )
        message(SEND_ERROR
                "${BoldRed} install_confix_python_tools:\n"
                "   Failed to install ConfiX python tools! ${ColourReset}\n"
                "   Error message:"
                "   ${_confixmsg}")

      endif()

      message(FATAL_ERROR
              "${BoldRed} install_genex_prebuilder_tools:\n"
              "   Failed to install GENE-X prebuilder tools! ${ColourReset}\n"
              "   Error message:"
              "   ${_errmsg}")
    else()
      message(WARNING
              "${BoldYellow} install_genex_prebuilder_tools:\n"
              "   Failed to install GENE-X prebuilder tools! ${ColourReset}\n"
              "   See prebuilder_pyinstall.err for the error log.\n"
              "   User may proceed building GENE-X.")
    endif()
  endif()

  # Unset temporary variables
  unset(_status)
  unset(_outmsg)
  unset(_errmsg)

endfunction()
