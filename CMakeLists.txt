cmake_minimum_required(VERSION 3.16)
set(SYCL_COMPILER "oneAPI" CACHE STRING "SYCL implementation to use: ComputeCpp|hipSYCL|triSYCL|oneAPI" FORCE)

#   oneAPI specific
if (${SYCL_COMPILER} STREQUAL "oneAPI")
    set(CMAKE_CXX_COMPILER dpcpp)
endif ()


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
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:CLANG>:-O0 -g3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
else ()
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-O3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:CLANG>:-O3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
endif ()

set(BUILD_SYCL ON)
option(BUILD_OCL OFF)
option(BUILD_OMP OFF)
option(BUILD_MPI OFF)


#  MODULES SETUP
#if (${BUILD_SYCL})
add_subdirectory(sycl)
#endif ()

if (${BUILD_OCL})
    add_subdirectory(ocl)
endif ()

if (${BUILD_OMP})
    add_subdirectory(omp)
endif ()

if (${BUILD_MPI})
    add_subdirectory(mpi)
endif ()


#   MISC
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach ()