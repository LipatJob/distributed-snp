#include <mpi.h>
#include <gtest/gtest.h>
#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"

enum class SimulatorBackend {
    NAIVE_CPU,
    NAIVE_CUDA_MPI,
    CUDA_MPI
};

// Factory wrapper for different backends
std::unique_ptr<ISnpSimulator> createSimulator(SimulatorBackend backend) {
    switch (backend) {
        case SimulatorBackend::NAIVE_CPU:
            return createNaiveCpuSimulator();
        // case SimulatorBackend::NAIVE_CUDA_MPI:
        //     return createNaiveCudaMpiSimulator();
        case SimulatorBackend::CUDA_MPI:
            return createCudaMpiSimulator();
        default:
            return nullptr;
    }
}

class SnpSimulatorTest : public ::testing::TestWithParam<SimulatorBackend> {
protected:
    std::unique_ptr<ISnpSimulator> s;
    
    void SetUp() override {
        s = createSimulator(GetParam());
    }
};

/**
 * @brief Test: One Spike Chain
 * Structure: [0] -> [1] -> [2] -> [3]
 * Logic: Spike passes one neuron per step.
 */
TEST_P(SnpSimulatorTest, OneSpikeChain) {
    
    auto config = SnpSystemBuilder()
        // Define Neurons: ID, Initial Spikes
        .addNeuron(0, 1) // Start with 1 spike
        .addNeuron(1, 0)
        .addNeuron(2, 0)
        .addNeuron(3, 0)
        // Define Rule: a -> a (Threshold 1, Consume 1, Produce 1)
        .addRule(0, 1, 1, 1)
        .addRule(1, 1, 1, 1)
        .addRule(2, 1, 1, 1)
        .addRule(3, 1, 1, 1)
        // Define Topology
        .chainNeurons({0, 1, 2, 3})
        .build();

    ASSERT_TRUE(s->loadSystem(config));

    // T=0: [1, 0, 0, 0]
    EXPECT_EQ(s->getLocalState(), std::vector<int>({1, 0, 0, 0}));

    // T=1: Spike moves 0->1
    s->step(); 
    EXPECT_EQ(s->getLocalState(), std::vector<int>({0, 1, 0, 0}));

    // T=2: Spike moves 1->2
    s->step();
    EXPECT_EQ(s->getLocalState(), std::vector<int>({0, 0, 1, 0}));

    // T=3: Spike moves 2->3
    s->step();
    EXPECT_EQ(s->getLocalState(), std::vector<int>({0, 0, 0, 1}));
}

/**
 * @brief Test: Consumption Logic
 * Logic: Rule a^3 / a^2 -> a consumes 2 spikes, leaving remainder.
 */
TEST_P(SnpSimulatorTest, ConsumptionLeavesRemainder) {

    auto config = SnpSystemBuilder()
        .addNeuron(0, 5) // Initial: 5 spikes
        .addNeuron(1, 0)
        // Rule: Requires 3, Consumes 2, Produces 1
        .addRule(0, 3, 2, 1) 
        .addSynapse(0, 1)
        .build();

    s->loadSystem(config);
    s->step();

    auto state = s->getLocalState();
    EXPECT_EQ(state[0], 3); // 5 - 2 = 3
    EXPECT_EQ(state[1], 1); // Received 1
}

/**
 * @brief Test: Rule Delay
 * Logic: Output is suspended for 'd' ticks.
 */
TEST_P(SnpSimulatorTest, RuleDelaySuspendsOutput) {

    auto config = SnpSystemBuilder()
        .addNeuron(0, 1)
        .addNeuron(1, 0)
        // Rule: a -> a; 2 (Delay = 2)
        .addRule(0, 1, 1, 1, 2) 
        .addSynapse(0, 1)
        .build();

    s->loadSystem(config);

    // T=0: Rule fires, neuron closes, spike is scheduled
    s->step(); 
    EXPECT_EQ(s->getLocalState()[1], 0); // Not arrived yet

    // T=1: Delay tick 1
    s->step(); 
    EXPECT_EQ(s->getLocalState()[1], 0); // Still waiting

    // T=2: Delay tick 2 (Spike arrives at end of step/start of next)
    s->step(); 
    EXPECT_EQ(s->getLocalState()[1], 1); // Arrived!
}

/**
 * @brief Test: Weighted Synapses
 * Logic: Spikes are multiplied by synapse weight.
 */
TEST_P(SnpSimulatorTest, WeightedSynapseMultipliesSpikes) {

    auto config = SnpSystemBuilder()
        .addNeuron(0, 1)
        .addNeuron(1, 0)
        .addRule(0, 1, 1, 1) // Produce 1
        .addSynapse(0, 1, 10) // Weight 10
        .build();

    s->loadSystem(config);
    s->step();

    EXPECT_EQ(s->getLocalState()[1], 10);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    SnpSimulatorTest,
    ::testing::Values(
        SimulatorBackend::NAIVE_CPU,
        // SimulatorBackend::NAIVE_CUDA_MPI,
        SimulatorBackend::CUDA_MPI
    ),
    [](const ::testing::TestParamInfo<SimulatorBackend>& info) {
        switch (info.param) {
            case SimulatorBackend::NAIVE_CPU: return "NaiveCPU";
            // case SimulatorBackend::NAIVE_CUDA_MPI: return "NaiveCudaMPI";
            case SimulatorBackend::CUDA_MPI: return "CudaMPI";
            default: return "Unknown";
        }
    }
);

int main(int argc, char** argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    // 2. Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Optional: Add a listener to print test results only from Rank 0
    // This prevents cluttered output where every rank prints "[  PASSED  ]"
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // 3. Run Tests
    int result = RUN_ALL_TESTS();

    // 4. Finalize MPI
    MPI_Finalize();

    return result;
}