#include <memory>

/**
 * @brief Interface for sorting integer arrays
 * 
 * This interface provides a contract for implementing various sorting algorithms
 * for integer data.
 */
class ISort {
public:
    virtual ~ISort() = default;

    /**
     * @brief Sort an array of integers in ascending order
     * 
     * @param data Pointer to the array of integers to sort
     * @param size Number of elements in the array
     */
    virtual void sort(int* data, size_t size) = 0;
};

std::unique_ptr<ISort> createNaiveCpuSnpSort();
std::unique_ptr<ISort> createCudaMpiSnpSort();
std::unique_ptr<ISort> createNaiveCudaMpiSnpSort();