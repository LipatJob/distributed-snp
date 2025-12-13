#include <memory>
#include <vector>

/**
 * @brief Interface for sorting integer arrays
 * 
 * This interface provides a contract for implementing various sorting algorithms
 * for integer data. It separates the loading/preparation phase from the actual
 * sorting execution to enable better benchmarking.
 */
class ISort {
public:
    virtual ~ISort() = default;

    /**
     * @brief Load and prepare data for sorting (not timed in benchmarks)
     * 
     * This method builds the necessary data structures and prepares the
     * sorting system. For SNP-based implementations, this includes building
     * the SNP system configuration and loading it into the simulator.
     * 
     * @param data Pointer to the array of integers to sort
     * @param size Number of elements in the array
     */
    virtual void load(const int* data, size_t size) = 0;

    /**
     * @brief Execute the sorting operation (timed in benchmarks)
     * 
     * This method performs the actual sorting computation. For SNP-based
     * implementations, this runs the simulation and extracts results.
     * 
     * @return Vector containing the sorted integers
     */
    virtual std::vector<int> execute() = 0;

    /**
     * @brief Sort an array of integers in ascending order (convenience method)
     * 
     * This is a convenience method that combines load() and execute() for
     * backward compatibility with existing code.
     * 
     * @param data Pointer to the array of integers to sort
     * @param size Number of elements in the array
     */
    virtual void sort(int* data, size_t size) {
        load(data, size);
        std::vector<int> result = execute();
        for (size_t i = 0; i < size && i < result.size(); ++i) {
            data[i] = result[i];
        }
    }

    /**
     * @brief Get performance metrics from the underlying implementation
     * 
     * For SNP-based implementations, this returns metrics like communication
     * time and compute time from the simulator.
     * 
     * @return String containing performance report
     */
    virtual std::string getPerformanceReport() const = 0;
};

std::unique_ptr<ISort> createNaiveCpuSnpSort();
std::unique_ptr<ISort> createCudaSnpSort();
std::unique_ptr<ISort> createSparseCudaSnpSort();
std::unique_ptr<ISort> createNaiveCudaMpiSnpSort();
std::unique_ptr<ISort> createCudaMpiSnpSort();