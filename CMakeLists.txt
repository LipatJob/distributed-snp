cmake_minimum_required(VERSION 3.18)
project(new-snp LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Option to enable/disable tests
option(BUILD_TESTS "Build tests" ON)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src/linear_algebra
    ${PROJECT_SOURCE_DIR}/src/snp
    ${MPI_CXX_INCLUDE_DIRS}
)

# ============================================================================
# Linear Algebra Library
# ============================================================================

# CPU implementation
add_library(cpu_matrix_ops STATIC
    src/linear_algebra/CpuMatrixOps.cpp
)
target_link_libraries(cpu_matrix_ops PUBLIC OpenMP::OpenMP_CXX)

# CUDA implementation
add_library(cuda_matrix_ops STATIC
    src/linear_algebra/CudaMatrixOps.cu
)
set_target_properties(cuda_matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # Adjust based on your GPU architecture
)
target_link_libraries(cuda_matrix_ops PUBLIC CUDA::cudart)

# MPI+CUDA implementation
add_library(mpi_cuda_matrix_ops STATIC
    src/linear_algebra/MpiCudaMatrixOps.cu
)
set_target_properties(mpi_cuda_matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # Adjust based on your GPU architecture
)
target_link_libraries(mpi_cuda_matrix_ops PUBLIC 
    CUDA::cudart
    MPI::MPI_CXX
)
target_include_directories(mpi_cuda_matrix_ops PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Combined matrix operations library
add_library(matrix_ops INTERFACE)
target_link_libraries(matrix_ops INTERFACE
    cpu_matrix_ops
    cuda_matrix_ops
    mpi_cuda_matrix_ops
)

# ============================================================================
# Main Executable
# ============================================================================

add_executable(matrix_demo
    src/main.cpp
)

target_link_libraries(matrix_demo PRIVATE
    matrix_ops
    MPI::MPI_CXX
)

# ============================================================================
# Tests (if enabled)
# ============================================================================

if(BUILD_TESTS)
    # Find GTest
    find_package(GTest)
    
    if(NOT GTest_FOUND)
        # Download and build GTest if not found
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    enable_testing()

    # Linear Algebra Tests
    add_executable(test_matrix_ops
        tests/linear_algebra/MatrixOps.cpp
    )
    
    target_link_libraries(test_matrix_ops PRIVATE
        matrix_ops
        GTest::gtest
        GTest::gtest_main
        MPI::MPI_CXX
    )

    # Use MPI to run tests (use 2 processes to avoid slot issues)
    add_test(NAME MatrixOpsTest 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --host localhost,10.0.0.2
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_matrix_ops> ${MPIEXEC_POSTFLAGS} --allow-run-as-root --oversubscribe)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS cpu_matrix_ops cuda_matrix_ops mpi_cuda_matrix_ops matrix_demo
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES src/linear_algebra/MatrixOps.h src/linear_algebra/KernelMatrixOps.cuh
    DESTINATION include
)

# ============================================================================
# Print configuration summary
# ============================================================================

message(STATUS "")
message(STATUS "==============================================")
message(STATUS "Configuration Summary:")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA Compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "  MPI Compiler:      ${MPI_CXX_COMPILER}")
message(STATUS "  OpenMP:            ${OpenMP_CXX_VERSION}")
message(STATUS "  CUDA Version:      ${CUDAToolkit_VERSION}")
message(STATUS "  Build Tests:       ${BUILD_TESTS}")
message(STATUS "==============================================")
message(STATUS "")
