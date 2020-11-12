cmake_minimum_required(VERSION 3.16)
set(BASE_NAME hello_world_sycl)
project(${BASE_NAME})


#   CMAKE & CXX SETUP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS ${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-O0 -g3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
else ()
    set(BASE_CXX_FLAGS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-O3 -pedantic -Wall -Wextra -Wno-unknown-pragmas -static-libstdc++>)
endif ()


#   TARGET INIT
set(SRC_FILES main.cpp)

add_executable(${BASE_NAME} ${SRC_FILES})

target_compile_options(${BASE_NAME} ${BASE_CXX_FLAGS})


#   SYCL SETUP
set(SYCL_IMPL "hipSYCL" CACHE STRING "SYCL implementation to use: ComputeCpp|hipSYCL" FORCE)

if (${SYCL_IMPL} STREQUAL "ComputeCpp")
    find_package(ComputeCpp REQUIRED)

    set(COMPUTECPP_USER_FLAGS
            -I/usr/include/c++/9
            -I/usr/include/x86_64-linux-gnu/c++/9
            -no-serial-memop
            )

    target_include_directories(${BASE_NAME}
            PRIVATE ${ComputeCpp_INCLUDE_DIRS}
            )

elseif (${SYCL_IMPL} STREQUAL "hipSYCL")
    set(hipSYCL_DIR "/usr/local/lib/cmake")

    find_package(hipSYCL CONFIG REQUIRED)

    set(HIPSYCL_PLATFORM cuda CACHE STRING "hipSYCL platform: cpu|rocm|cuda" FORCE)
    set(HIPSYCL_GPU_ARCH sm_30 CACHE STRING "hipSYCL GPU architecture, eg. gfx803" FORCE)

    target_include_directories(${BASE_NAME}
            PUBLIC /usr/local/include/hipSYCL/contrib
            )

    set(HIPSYCL_SYCLCC_LAUNCH_RULE "${HIPSYCL_SYCLCC_LAUNCH_RULE} -O2 ")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(HIPSYCL_DEBUG_LEVEL=3)
    else ()
        add_compile_definitions(HIPSYCL_DEBUG_LEVEL=1)
    endif ()

else ()
    message(FATAL_ERROR "Wrong or not supported SYCL implementation: " ${PNDFTSTR_SYCL_IMPL})
endif ()

add_sycl_to_target(TARGET ${BASE_NAME} SOURCES ${SRC_FILES})


#   MISC
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach ()