cmake_minimum_required(VERSION 3.18)
project(new-snp LANGUAGES CXX CUDA)

# ============================================================================
# 1. Project Configuration & Defaults
# ============================================================================

# C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configuration Variables (Easily editable)
set(CUDA_ARCH_TARGET "75" CACHE STRING "Target CUDA architecture")
set(MPI_TEST_HOSTS "localhost,10.0.0.2" CACHE STRING "Hosts for MPI tests")
option(BUILD_TESTS "Build tests and benchmarks" ON)

# --- KEY REQUEST: Centralized Build Artifacts ---
# This ensures all binaries/libs go to build/bin and build/lib immediately
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Global Compiler Warnings (Applied to C++ targets)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g -O0>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
    add_compile_definitions(NDEBUG)
endif()

# ============================================================================
# 2. Dependencies
# ============================================================================

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# 3. SNP System Library
# ============================================================================

add_library(snp_system STATIC
    src/snp/SnpSystemConfig.hpp
    src/snp/ISnpSimulator.hpp
    src/snp/NaiveCpuSnpSimulator.cpp
    src/snp/OptimizedCudaSnpSimulator.cu
    src/snp/SparseCudaSnpSimulator.cu
    src/snp/NaiveCudaMpiSnpSimulator.cu
    src/snp/OptimizedCudaMpiSnpSimulator.cu
    src/partitioner/IPartitioner.hpp
    src/partitioner/LinearPartitioner.hpp
    src/partitioner/LouvainPartitioner.cpp
    src/partitioner/RedBluePartitioner.hpp
)

set_target_properties(snp_system PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CUDA_ARCH_TARGET}
)

target_include_directories(snp_system PUBLIC ${PROJECT_SOURCE_DIR}/src/snp)
target_include_directories(snp_system PUBLIC ${PROJECT_SOURCE_DIR}/src/partitioner)
target_link_libraries(snp_system PUBLIC MPI::MPI_CXX CUDA::cudart)

# ============================================================================
# 4. SNP Sort Library
# ============================================================================

add_library(snp_sort STATIC src/sort/SnpSort.cpp)
target_include_directories(snp_sort PUBLIC ${PROJECT_SOURCE_DIR}/src/sort)
target_link_libraries(snp_sort PUBLIC snp_system)

# ============================================================================
# 5. Main Executables
# ============================================================================

# Profiling
add_executable(snp_profile profiling/SnpProfile.cpp)
target_link_libraries(snp_profile PRIVATE snp_sort snp_system MPI::MPI_CXX)

# ============================================================================
# 7. Tests & Benchmarks
# ============================================================================

if(BUILD_TESTS)
    include(FetchContent)
    enable_testing()

    # --- Fetch Dependencies ---
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )
    FetchContent_MakeAvailable(googletest)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    FetchContent_MakeAvailable(googlebenchmark)

    # --- Helper Macro for MPI Tests ---
    # Reduces repetition for adding MPI tests
    macro(add_mpi_test test_name target_name)
        add_test(NAME ${test_name}
                 COMMAND ${MPIEXEC_EXECUTABLE} 
                 ${MPIEXEC_NUMPROC_FLAG} 2 
                 --host ${MPI_TEST_HOSTS}
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target_name}> ${MPIEXEC_POSTFLAGS} 
                 --allow-run-as-root)
    endmacro()

    # --- Define Tests ---
    
    # SNP Simulator
    add_executable(test_snp_simulator tests/snp/SnpSimulatorTest.cpp)
    target_link_libraries(test_snp_simulator PRIVATE snp_system GTest::gtest GTest::gtest_main MPI::MPI_CXX)
    add_mpi_test(SnpSimulatorTest test_snp_simulator)

    # SNP Sort
    add_executable(test_sort tests/sort/SnpSortTest.cpp)
    target_link_libraries(test_sort PRIVATE snp_sort snp_system GTest::gtest GTest::gtest_main MPI::MPI_CXX)
    add_mpi_test(SnpSortTest test_sort)

    # Partitioner
    add_executable(test_partitioner tests/partitioner/PartitionerTest.cpp)
    target_link_libraries(test_partitioner PRIVATE snp_system GTest::gtest GTest::gtest_main MPI::MPI_CXX)
    add_mpi_test(PartitionerTest test_partitioner)
    
    # --- Benchmarks ---
    add_executable(sort_benchmark benchmark/SortBenchmark.cpp)
    target_link_libraries(sort_benchmark PRIVATE snp_sort snp_system benchmark::benchmark benchmark::benchmark_main MPI::MPI_CXX)

    # --- Microbenchmark Tool ---
    add_executable(microbenchmark benchmark/Microbenchmark.cpp)
    target_link_libraries(microbenchmark PRIVATE snp_sort snp_system MPI::MPI_CXX)
endif()

# ============================================================================
# 8. Installation
# ============================================================================

# Note: CMAKE_RUNTIME_OUTPUT_DIRECTORY handles the build folder structure,
# This handles the `make install` structure.
install(TARGETS snp_system snp_sort
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(BUILD_TESTS)
    install(TARGETS snp_profile sort_benchmark RUNTIME DESTINATION bin)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "-------------------------------------------------")
message(STATUS "Project:             ${PROJECT_NAME}")
message(STATUS "Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "Output Directory:    ${CMAKE_BINARY_DIR}/bin")
message(STATUS "CUDA Architecture:   ${CUDA_ARCH_TARGET}")
message(STATUS "MPI Test Hosts:      ${MPI_TEST_HOSTS}")
message(STATUS "-------------------------------------------------")