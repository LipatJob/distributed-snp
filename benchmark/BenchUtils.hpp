#include <random>

namespace BenchUtils {

    enum class Distribution {
        RANDOM, SORTED, REVERSE_SORTED, NEARLY_SORTED, FEW_UNIQUE, UNIFORM
    };

    struct TestConfig {
        std::string name;
        size_t size;
        int maxVal;
        Distribution dist;
        int iterations = 0; // 0 = default, 1 = forced (needed for heavy MPI)
    };

    std::string DistToString(Distribution d) {
        switch(d) {
            case Distribution::RANDOM: return "Random";
            case Distribution::SORTED: return "Sorted";
            case Distribution::REVERSE_SORTED: return "Reverse";
            case Distribution::NEARLY_SORTED: return "NearlySorted";
            case Distribution::FEW_UNIQUE: return "FewUnique";
            case Distribution::UNIFORM: return "Uniform";
            default: return "Unknown";
        }
    }

    std::vector<int> GenerateData(size_t size, int maxValue, Distribution dist, unsigned seed) {
        std::vector<int> data(size);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> valDist(0, maxValue);

        // (Keeping generation logic compact for brevity - insert your full logic here)
        switch (dist) {
            case Distribution::SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end());
                break;
            case Distribution::REVERSE_SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end(), std::greater<int>());
                break;
            case Distribution::UNIFORM:
                std::fill(data.begin(), data.end(), maxValue);
                break;
            case Distribution::NEARLY_SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end());
                for (size_t i = 0; i < size / 10; ++i) {
                    size_t idx1 = rng() % size;
                    size_t idx2 = rng() % size;
                    std::swap(data[idx1], data[idx2]);
                }
                break;
            case Distribution::FEW_UNIQUE: {
                int uniqueCount = std::max(2, maxValue / 10);
                std::vector<int> uniqueValues(uniqueCount);
                for (auto& x : uniqueValues) x = valDist(rng) % maxValue;
                for (auto& x : data) x = uniqueValues[rng() % uniqueCount];
                break;
            }
            case Distribution::RANDOM: {
                for(auto& x : data) x = valDist(rng);
                break;
            }
            default: // Random and others
                for(auto& x : data) x = valDist(rng);
                break;
        }
        return data;
    }

    bool IsSorted(const std::vector<int>& data) {
        return std::is_sorted(data.begin(), data.end());
    }
}
