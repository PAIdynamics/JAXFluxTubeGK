#[=======================================================================[.rst:
LoadConfiX
----------

.. version:: 1.0

This macro finds, manages and loads the corect path of ConfiX for more
consistency accross different projects.
Copied from: <ConfiX>/cmake/templates/LoadConfiX.cmake

.. _`basic signature`:

Basic Signature
^^^^^^^^^^^^^^^

.. parsed-literal::

  load_confix([DEFAULT_PATH] <path>)

Mandatory argument
^^^^^^^^^^^^^^^^^^

``DEFAULT_PATH``
    A string argument specifying the default path to ConfiX.

Optional cache variable
^^^^^^^^^^^^^^^^^^^^^^^

``ConfiX_DIR``
    A string argument specifying the user-specified path to ConfiX.
    With `ConfiX_DIR`, user is able to set the path to out-of-source ConfiX.

#]=======================================================================]

macro(load_confix)
  # List of names of the boolean arguments (only defined ones will be true)
  set(_OPTIONS_ARGS)
  # List of names of single-valued arguments
  set(_ONE_VALUE_ARGS "DEFAULT_PATH")
  # List of names of multi-valued arguments
  set(_MULTI_VALUE_ARGS)

  # Parse the arguments
  cmake_parse_arguments(_PARSED_ARGS
    "${_OPTIONS_ARGS}" "${_ONE_VALUE_ARGS}" "${_MULTI_VALUE_ARGS}" ${ARGN})

  # Check mandatory argument: "DEFAULT_PATH"
  if(NOT DEFINED _PARSED_ARGS_DEFAULT_PATH)
    message(FATAL_ERROR
      "load_confix: \"DEFAULT_PATH\" argument is required!")
  endif()

  # Set ConfiX path to the default path
  set(_confix_path ${_PARSED_ARGS_DEFAULT_PATH})

  # Check ConfiX is available in the top-level project
  if(EXISTS "${CMAKE_SOURCE_DIR}/confix/cmake")
    set(_confix_path "${CMAKE_SOURCE_DIR}/confix/cmake")
  endif()

  # ConfiX_DIR overrides the path if available
  if(DEFINED ConfiX_DIR)
    set(_confix_path ${ConfiX_DIR})
  endif()

  # Load ConfiX path to CMAKE module paths
  list(APPEND CMAKE_MODULE_PATH ${_confix_path})

  string(ASCII 27 _esc)
  set(_bgreen "${_esc}[1;32m")
  set(_creset "${_esc}[m")
  message(STATUS "${_bgreen}Use ConfiX: ${_confix_path}${_creset}")

  # Unset temporary variables
  unset(_OPTIONS_ARGS)
  unset(_ONE_VALUE_ARGS)
  unset(_MULTI_VALUE_ARGS)
  unset(_PARSED_ARGS_DEFAULT_PATH)
  unset(_confix_path)
  unset(_esc)
  unset(_bgreen)
  unset(_creset)

endmacro()
