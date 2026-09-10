# The Python bindings.
#
# Reached only from the top level under ASH_BUILD_PYTHON, which is off by
# default. Everything that needs Python -- the interpreter, the headers, the
# pybind11 fetch -- therefore lives inside this file rather than in cmake/deps,
# because a fetch that ran unconditionally would take down every CI job that does
# not install python3-devel before it compiled a single line.

# Interpreter first, and before pybind11 is made available. pybind11 runs its own
# find_package(Python) while configuring; finding it here means that call lands on
# the cached result and the module is built for the same interpreter this file
# reports, which matters on a machine where 3.13 and 3.14 are both installed and
# a module built for one will not import into the other.
#
# Development.Module and not Development or Development.Embed: a Linux extension
# does not link libpython -- its symbols are resolved by the interpreter that
# dlopens it -- and asking for the embedding component would demand a shared
# libpython that a conda base environment does not have.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

message(STATUS "ash python bindings: interpreter ${Python_EXECUTABLE} (${Python_VERSION})")

include(FetchContent)

# 3.0 is the floor because Python 3.14 support starts there, and Fedora 43 ships
# 3.14 as its system Python -- which is what the CI container has.
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.4
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core
    module.cpp
    convert.cpp
    types.cpp
    provider.cpp
    tools.cpp
    run.cpp)

target_link_libraries(_core PRIVATE ash_core ash_eval)

# Deliberately not ash_set_sanitizers: see the note on ASH_BUILD_PYTHON.
ash_set_warnings(_core)

# _core has to land next to __init__.py, and the package has to be the directory
# on PYTHONPATH, so both go one level down. No SOABI suffix is written anywhere --
# pybind11_add_module sets it from the interpreter found above, and hardcoding
# one would be wrong on the other machine.
set(ASH_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ash")

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${ASH_PYTHON_PACKAGE_DIR}")

# The package's Python half, copied whenever it changes.
#
# As an OUTPUT with its source as a dependency, not as a POST_BUILD step on the
# module and not as file(COPY): a POST_BUILD step only runs when the extension
# relinks, so editing __init__.py on its own would leave the build tree quietly
# importing the old one, and file(COPY) defers the same surprise to whenever
# someone next reconfigures.
add_custom_command(
    OUTPUT "${ASH_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ASH_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ash/__init__.py"
            "${ASH_PYTHON_PACKAGE_DIR}/__init__.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ash/__init__.py"
    COMMENT "Copying ash/__init__.py into the build tree")

add_custom_target(ash_python_package ALL
    DEPENDS "${ASH_PYTHON_PACKAGE_DIR}/__init__.py")

# So that tools/verify_python.sh can run the tests under the same interpreter the
# module was built for instead of whatever python3 happens to resolve to.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/python-interpreter.txt"
     "${Python_EXECUTABLE}\n")
