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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(deps)
include(warnings)
include(sanitizers)

option(ASH_BUILD_TESTS "Build ash tests" ON)
option(ASH_BUILD_BENCHMARKS "Build ash benchmarks" OFF)
option(ASH_BUILD_CLI "Build the ash command-line tool" ON)
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
