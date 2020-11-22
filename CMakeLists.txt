cmake_minimum_required(VERSION 3.16)
project(hello_worl LANGUAGES CXX)

#   CMAKE & CXX SETUP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "CMake build type: " ${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-O0 -g3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
else ()
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-O3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
endif ()

option(BUILD_SYCL ON)
option(BUILD_OCL OFF)
option(BUILD_OMP OFF)
option(BUILD_MPI OFF)


#  MODULES SETUP
if (${BUILD_SYCL})
    include_directories(sycl)
endif ()

if (${BUILD_OCL})
include_directories(ocl)
endif ()

if (${BUILD_OMP})
include_directories(omp)
endif ()

if (${BUILD_MPI})
include_directories(mpi)
endif ()


#   MISC
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach ()