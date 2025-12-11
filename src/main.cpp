#include "MatrixOps.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// Helper function to print a matrix
template <typename T>
void printMatrix(const string& name, const vector<T>& mat, size_t rows, size_t cols, size_t maxDisplay = 5) {
    cout << name << " (" << rows << "x" << cols << "):\n";
    size_t displayRows = min(rows, maxDisplay);
    size_t displayCols = min(cols, maxDisplay);
    
    for (size_t i = 0; i < displayRows; ++i) {
        cout << "  ";
        for (size_t j = 0; j < displayCols; ++j) {
            cout << fixed << setprecision(2) << setw(8) << mat[i * cols + j] << " ";
        }
        if (cols > maxDisplay) cout << "...";
        cout << "\n";
    }
    if (rows > maxDisplay) cout << "  ...\n";
    cout << "\n";
}

// Helper function to initialize a matrix with values
template <typename T>
void initializeMatrix(vector<T>& mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<T>(i % 10) * 0.1f;
    }
}

// Helper function to verify results match
template <typename T>
bool verifyResults(const vector<T>& result1, const vector<T>& result2, T tolerance = 1e-3) {
    if (result1.size() != result2.size()) return false;
    for (size_t i = 0; i < result1.size(); ++i) {
        if (abs(result1[i] - result2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Test a specific backend
template <typename T>
void testBackend(BackendType backend, const vector<T>& A, const vector<T>& B, 
                 vector<T>& C, size_t m, size_t n, size_t k, int rank) {
    auto ops = createMatrixOps<T>(backend);
    
    if (rank == 0) {
        cout << "Testing " << ops->getBackendName() << " backend...\n";
    }
    
    // Warmup run
    ops->multiply(A.data(), B.data(), C.data(), m, n, k);
    
    // Timed run
    auto start = high_resolution_clock::now();
    ops->multiply(A.data(), B.data(), C.data(), m, n, k);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    if (rank == 0) {
        cout << "  Execution time: " << duration.count() / 1000.0 << " ms\n";
        printMatrix("Result C", C, m, n);
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Matrix dimensions (adjust as needed)
    const size_t m = 512;  // Rows of A and C
    const size_t k = 512;  // Cols of A, Rows of B
    const size_t n = 512;  // Cols of B and C
    
    if (rank == 0) {
        cout << "=======================================================\n";
        cout << "Matrix Multiplication Demo\n";
        cout << "=======================================================\n";
        cout << "Matrix dimensions:\n";
        cout << "  A: " << m << " x " << k << "\n";
        cout << "  B: " << k << " x " << n << "\n";
        cout << "  C: " << m << " x " << n << "\n";
        cout << "MPI Processes: " << size << "\n";
        cout << "=======================================================\n\n";
    }
    
    // Initialize matrices
    vector<float> A(m * k);
    vector<float> B(k * n);
    vector<float> C_cpu(m * n, 0.0f);
    vector<float> C_cuda(m * n, 0.0f);
    vector<float> C_mpi_cuda(m * n, 0.0f);
    
    initializeMatrix(A, m, k);
    initializeMatrix(B, k, n);
    
    if (rank == 0) {
        printMatrix("Input A", A, m, k);
        printMatrix("Input B", B, k, n);
    }
    
    // Test CPU backend
    testBackend(BackendType::CPU, A, B, C_cpu, m, n, k, rank);
    
    // Test CUDA backend
    testBackend(BackendType::CUDA, A, B, C_cuda, m, n, k, rank);
    
    // Test MPI+CUDA backend
    testBackend(BackendType::MPI_CUDA, A, B, C_mpi_cuda, m, n, k, rank);
    
    // Verify results match
    if (rank == 0) {
        cout << "=======================================================\n";
        cout << "Verification:\n";
        bool cpuCudaMatch = verifyResults(C_cpu, C_cuda);
        bool cpuMpiMatch = verifyResults(C_cpu, C_mpi_cuda);
        bool cudaMpiMatch = verifyResults(C_cuda, C_mpi_cuda);
        
        cout << "  CPU vs CUDA:     " << (cpuCudaMatch ? "✓ MATCH" : "✗ MISMATCH") << "\n";
        cout << "  CPU vs MPI+CUDA: " << (cpuMpiMatch ? "✓ MATCH" : "✗ MISMATCH") << "\n";
        cout << "  CUDA vs MPI+CUDA: " << (cudaMpiMatch ? "✓ MATCH" : "✗ MISMATCH") << "\n";
        
        if (cpuCudaMatch && cpuMpiMatch && cudaMpiMatch) {
            cout << "\n✓ All backends produced identical results!\n";
        } else {
            cout << "\n✗ Warning: Results differ between backends!\n";
        }
        cout << "=======================================================\n";
    }
    
    MPI_Finalize();
    return 0;
}
