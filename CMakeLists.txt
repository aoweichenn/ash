cmake_minimum_required(VERSION 3.25)

project(ash
    VERSION 0.1.0
    DESCRIPTION "A deterministic, observable agent runtime in C++20"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Global and before any target exists, because the Python extension has to link
# both of the static libraries and everything they carry. Two things force it:
# the runtime's polymorphic interfaces put vtable pointers in data, which is an
# absolute relocation a shared object cannot have, and ash_core names spdlog as a
# private dependency, which CMake passes on as LINK_ONLY -- so libspdlog.a ends
# up on the module's link line even though no header includes it today. Setting
# this per target would leave the second one as a trap for whoever includes a
# spdlog header first. Fedora already compiles PIE by default, so the cost to the
# existing test builds is nothing.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(deps)
include(warnings)
include(sanitizers)

option(ASH_BUILD_TESTS "Build ash tests" ON)
option(ASH_BUILD_BENCHMARKS "Build ash benchmarks" OFF)
option(ASH_BUILD_CLI "Build the ash command-line tool" ON)

# Off by default: a C++-only checkout should not need Python headers, and gating
# the pybind11 fetch behind it is what keeps the CI jobs that never install them
# configuring as before. The sanitizer presets pin it off too -- the extension
# would have to be instrumented alongside the runtime or it fails to import on an
# unresolved __asan_init, and CPython then reports as one large leak.
option(ASH_BUILD_PYTHON "Build the Python bindings" OFF)
option(ASH_WERROR "Treat compiler warnings as errors" OFF)

add_subdirectory(src)

# The eval harness is a consumer of ash_core, kept out of include/ash so that
# embedding the runtime does not mean compiling a suite parser and a price
# table. A test enforces the boundary.
add_subdirectory(eval)

if(ASH_BUILD_CLI)
    add_subdirectory(apps/cli)
endif()

if(ASH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Off by default and reached only from here, so that a checkout without Python
# development headers -- which includes every job in the CI matrix except the one
# that installs them -- configures and builds exactly as it did before.
if(ASH_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()
