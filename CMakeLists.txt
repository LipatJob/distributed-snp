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
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"  # Adjust based on your GPU architecture
)
target_link_libraries(cuda_matrix_ops PUBLIC CUDA::cudart)

# MPI+CUDA implementation
add_library(mpi_cuda_matrix_ops STATIC
    src/linear_algebra/MpiCudaMatrixOps.cu
)
set_target_properties(mpi_cuda_matrix_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"  # Adjust based on your GPU architecture
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
# SNP System Library
# ============================================================================

add_library(snp_system STATIC
    src/snp/SnpSystemConfig.hpp
    src/snp/ISnpSimulator.hpp
    src/snp/NaiveCpuSnpSimulator.cpp
    src/snp/CudaSnpSimulator.cu
    src/snp/NaiveCudaMpiSnpSimulator.cu
    src/snp/CudaMpiSnpSimulator.cu
)
set_target_properties(snp_system PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"  # Adjust based on your GPU architecture
)
target_link_libraries(snp_system PUBLIC
    matrix_ops
    MPI::MPI_CXX
    CUDA::cudart
)
target_include_directories(snp_system PUBLIC
    ${PROJECT_SOURCE_DIR}/src/snp
    ${PROJECT_SOURCE_DIR}/src/linear_algebra
    ${MPI_CXX_INCLUDE_DIRS}
)

# ============================================================================
# SNP Sort Library
# ============================================================================

add_library(snp_sort STATIC
    src/sort/SnpSort.cpp

)
target_link_libraries(snp_sort PUBLIC
    snp_system
)
target_include_directories(snp_sort PUBLIC
    ${PROJECT_SOURCE_DIR}/src/sort
    ${PROJECT_SOURCE_DIR}/src/snp
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
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    # Download and build Google Benchmark
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Enable installation of benchmark.")
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    FetchContent_MakeAvailable(googlebenchmark)

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

    # SNP Simulator Tests
    add_executable(test_snp_simulator
        tests/snp/SnpSimulator.cpp
    )
    
    target_link_libraries(test_snp_simulator PRIVATE
        snp_system
        matrix_ops
        GTest::gtest
        GTest::gtest_main
        MPI::MPI_CXX
    )

    # Use MPI to run SNP simulator tests
    add_test(NAME SnpSimulatorTest
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --host localhost,10.0.0.2
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_snp_simulator> ${MPIEXEC_POSTFLAGS} --allow-run-as-root --oversubscribe)

    # SNP Sort Tests
    add_executable(test_sort
        tests/sort/SnpSort.cpp
    )
    
    target_link_libraries(test_sort PRIVATE
        snp_sort
        snp_system
        GTest::gtest
        GTest::gtest_main
        MPI::MPI_CXX
    )

    # Use MPI to run SNP sort tests
    add_test(NAME SnpSortTest
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --host localhost,10.0.0.2
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_sort> ${MPIEXEC_POSTFLAGS} --allow-run-as-root --oversubscribe)
    
    # ============================================================================
    # Benchmarks
    # ============================================================================
    
    # Sort Benchmark
    add_executable(sort_benchmark
        benchmark/SortBenchmark.cpp
    )
    
    target_link_libraries(sort_benchmark PRIVATE
        snp_sort
        snp_system
        benchmark::benchmark
        benchmark::benchmark_main
        MPI::MPI_CXX
    )
    
    # Install benchmark executable
    install(TARGETS sort_benchmark
        RUNTIME DESTINATION bin
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS cpu_matrix_ops cuda_matrix_ops mpi_cuda_matrix_ops snp_system snp_sort matrix_demo
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
