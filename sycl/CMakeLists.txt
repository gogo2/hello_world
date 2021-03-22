set(BASE_NAME hello_world_sycl)



#   ComputeCpp specific
if (${SYCL_COMPILER} STREQUAL "ComputeCpp")
    add_compile_definitions(SYCL_COMPILER_COMPUTECPP)
endif ()

#   hipSYCL SPECIFIC
if (${SYCL_COMPILER} STREQUAL "hipSYCL")
    add_compile_definitions(SYCL_COMPILER_HIPSYCL)

    set(HIPSYCL_PLATFORM cpu CACHE STRING "hipSYCL platform: cpu|rocm|cuda" FORCE)
    set(HIPSYCL_GPU_ARCH sm_30 CACHE STRING "hipSYCL GPU architecture, eg. gfx803 or sm_30" FORCE)
endif ()

#   oneAPI specific
if (${SYCL_COMPILER} STREQUAL "oneAPI")
    add_compile_definitions(SYCL_COMPILER_ONEAPI)
endif ()

#   triSYCL specific
if (${SYCL_COMPILER} STREQUAL "triSYCL")
    add_compile_definitions(SYCL_COMPILER_TRISYCL)
endif ()

project(${BASE_NAME} LANGUAGES CXX)


#   TARGET INIT
set(SRC_FILES main.cpp)

add_executable(${BASE_NAME} ${SRC_FILES})

target_compile_options(${BASE_NAME} ${BASE_CXX_FLAGS})


#   SYCL SETUP


if (${SYCL_COMPILER} STREQUAL "ComputeCpp")
    find_package(ComputeCpp REQUIRED)

    set(COMPUTECPP_USER_FLAGS
            -I/usr/include/c++/9
            -I/usr/include/x86_64-linux-gnu/c++/9
            -no-serial-memop
            )

    target_include_directories(${BASE_NAME}
            PRIVATE ${ComputeCpp_INCLUDE_DIRS}
            )

    add_sycl_to_target(TARGET ${BASE_NAME} SOURCES ${SRC_FILES})

elseif (${SYCL_COMPILER} STREQUAL "hipSYCL")
    set(hipSYCL_DIR "/usr/local/lib/cmake")

    find_package(hipSYCL CONFIG REQUIRED)

    target_include_directories(${BASE_NAME}
            PUBLIC /usr/local/include/hipSYCL/contrib
            )

    set(HIPSYCL_SYCLCC_LAUNCH_RULE "${HIPSYCL_SYCLCC_LAUNCH_RULE} -O3")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(HIPSYCL_DEBUG_LEVEL=3)
    else ()
        add_compile_definitions(HIPSYCL_DEBUG_LEVEL=1)
    endif ()

    add_sycl_to_target(TARGET ${BASE_NAME} SOURCES ${SRC_FILES})

elseif (${SYCL_COMPILER} STREQUAL "oneAPI")
elseif (${SYCL_COMPILER} STREQUAL "triSYCL")
    message(FATAL_ERROR "triSYCL is not supported yet")

else ()
    message(FATAL_ERROR "Wrong or not supported SYCL implementation: " ${SYCL_COMPILER})
endif ()

