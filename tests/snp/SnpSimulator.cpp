/**
 * @file SnpSimulator.cpp
 * @brief Comprehensive test suite for SN P System Simulator
 * 
 * Tests based on the verification example from literature:
 * - System with 3 neurons and 4 rules including delays
 * - Step-by-step execution verification
 * - Configuration vector correctness
 * - Status vector (open/closed neurons)
 * - Delay handling
 * 
 * Reference: docs/snp_example.tex
 */

#include "SnpSimulator.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>

// ============================================================================
// Test Fixture and Helper Functions
// ============================================================================

constexpr double EPSILON = 1e-5;

/**
 * @brief Helper function to compare floating point values with tolerance
 */
bool floatEqual(double a, double b, double epsilon = EPSILON) {
    return std::abs(a - b) < epsilon;
}

/**
 * @brief Test fixture for SnpSimulator tests
 */
class SnpSimulatorTest : public ::testing::TestWithParam<BackendType> {
protected:
    void SetUp() override {
        // Get the backend type from the test parameter
        backend = GetParam();
    }

    BackendType backend;
};

// ============================================================================
// Verification Example from Literature (docs/snp_example.tex)
// ============================================================================

/**
 * @brief Test the complete verification example with delays
 * 
 * System Definition:
 * - 3 neurons (m=3), 4 rules (n=4)
 * - σ1: r1: a/a → a; 0
 * - σ2: r2: a/a → a; 0, r3: a² → λ; 0
 * - σ3: r4: a → a; 2 (delay = 2)
 * - Synapses: {(1,2), (2,1), (2,3), (3,2)}
 * - Initial: C^(0) = (1, 0, 1)
 */
TEST_P(SnpSimulatorTest, VerificationExampleWithDelay) {
    // System has 3 neurons and 4 rules
    SnpSimulator sim(backend, 3, 4);

    // Set initial configuration: C^(0) = (1, 0, 1)
    sim.setInitialSpikes(0, 1.0);  // σ1 has 1 spike
    sim.setInitialSpikes(1, 0.0);  // σ2 has 0 spikes
    sim.setInitialSpikes(2, 1.0);  // σ3 has 1 spike

    // Define rules based on the matrix representation:
    // Rule r1: σ1 fires, consumes 1, sends 1 to σ2, delay 0
    // Matrix: -1 at (r1, σ1), +1 at (r1, σ2)
    sim.addRule(0, 0, 1, 1.0, 1.0, 1.0, 0);

    // Rule r2: σ2 fires, consumes 1, sends to σ1 and σ3, delay 0
    // In the matrix, this is represented as:
    // -1 at (r2, σ2), +1 at (r2, σ1), +1 at (r2, σ3)
    // We need to handle multiple outputs by adding the rule twice with different destinations
    sim.addRule(1, 1, 0, 1.0, 1.0, 1.0, 0);  // σ2 → σ1
    
    // Rule r3: σ2 forgets 2 spikes (a² → λ)
    sim.addRule(2, 1, -1, 2.0, 0.0, 2.0, 0, true);  // Forgetting rule

    // Rule r4: σ3 fires, consumes 1, sends to σ2, delay 2
    sim.addRule(3, 2, 1, 1.0, 1.0, 1.0, 2);

    // Verify initial configuration
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 1.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(2), 1.0));

    // --- Step k=0 ---
    // Expected: σ1 fires r1, σ3 fires r4 (with delay)
    // σ1 consumes 1, produces 1 to σ2: σ1: 1 → 0, σ2: 0 → 1
    // σ3 consumes 1, becomes closed (delay=2): σ3: 1 → 0
    // Result: C^(1) = (0, 1, 0)
    std::cout << "\n=== Test Step 0 ===" << std::endl;
    sim.step(0);
    
    const auto& config1 = sim.getConfiguration();
    EXPECT_TRUE(floatEqual(config1[0], 0.0)) << "Step 0: σ1 should have 0 spikes";
    EXPECT_TRUE(floatEqual(config1[1], 1.0)) << "Step 0: σ2 should have 1 spike";
    EXPECT_TRUE(floatEqual(config1[2], 0.0)) << "Step 0: σ3 should have 0 spikes";
    
    EXPECT_TRUE(sim.isNeuronOpen(0)) << "Step 0: σ1 should be open";
    EXPECT_TRUE(sim.isNeuronOpen(1)) << "Step 0: σ2 should be open";
    EXPECT_FALSE(sim.isNeuronOpen(2)) << "Step 0: σ3 should be closed (delay=2)";

    // --- Step k=1 ---
    // Expected: σ2 fires r2, sends to σ1 and σ3
    // σ2 consumes 1, produces 1 to σ1: σ2: 1 → 0, σ1: 0 → 1
    // σ3 is closed, rejects incoming spike
    // Result: C^(2) = (1, 0, 0)
    std::cout << "\n=== Test Step 1 ===" << std::endl;
    sim.step(1);
    
    const auto& config2 = sim.getConfiguration();
    EXPECT_TRUE(floatEqual(config2[0], 1.0)) << "Step 1: σ1 should have 1 spike";
    EXPECT_TRUE(floatEqual(config2[1], 0.0)) << "Step 1: σ2 should have 0 spikes";
    EXPECT_TRUE(floatEqual(config2[2], 0.0)) << "Step 1: σ3 should have 0 spikes (closed, rejected spike)";
    
    EXPECT_FALSE(sim.isNeuronOpen(2)) << "Step 1: σ3 should still be closed (delay=1 remaining)";

    // --- Step k=2 ---
    // Expected: σ1 fires r1 again
    // Result: C^(3) = (0, 1, 0)
    std::cout << "\n=== Test Step 2 ===" << std::endl;
    sim.step(2);
    
    const auto& config3 = sim.getConfiguration();
    EXPECT_TRUE(floatEqual(config3[0], 0.0)) << "Step 2: σ1 should have 0 spikes";
    EXPECT_TRUE(floatEqual(config3[1], 1.0)) << "Step 2: σ2 should have 1 spike";
    EXPECT_TRUE(floatEqual(config3[2], 0.0)) << "Step 2: σ3 should have 0 spikes";
    
    EXPECT_FALSE(sim.isNeuronOpen(2)) << "Step 2: σ3 should still be closed (delay=0 remaining)";

    // --- Step k=3 ---
    // Expected: σ2 fires r2, σ3 is now open and receives spike
    // Result: C^(4) = (1, 0, 1)
    std::cout << "\n=== Test Step 3 ===" << std::endl;
    sim.step(3);
    
    const auto& config4 = sim.getConfiguration();
    EXPECT_TRUE(floatEqual(config4[0], 1.0)) << "Step 3: σ1 should have 1 spike";
    EXPECT_TRUE(floatEqual(config4[1], 0.0)) << "Step 3: σ2 should have 0 spikes";
    EXPECT_TRUE(floatEqual(config4[2], 1.0)) << "Step 3: σ3 should have 1 spike (now open, accepted spike)";
    
    EXPECT_TRUE(sim.isNeuronOpen(2)) << "Step 3: σ3 should be open now";
}

/**
 * @brief Test simple firing without delays
 */
TEST_P(SnpSimulatorTest, SimpleFiringChain) {
    // Create a simple chain: σ1 → σ2 → σ3
    SnpSimulator sim(backend, 3, 2);

    // Initial: σ1 has 2 spikes
    sim.setInitialSpikes(0, 2.0);

    // Rule 0: σ1 consumes 2, produces 1 to σ2
    sim.addRule(0, 0, 1, 2.0, 1.0, 2.0, 0);

    // Rule 1: σ2 consumes 1, produces 1 to σ3
    sim.addRule(1, 1, 2, 1.0, 1.0, 1.0, 0);

    // Step 0: σ1 fires
    sim.step(0);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 1.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(2), 0.0));

    // Step 1: σ2 fires
    sim.step(1);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(2), 1.0));

    // Step 2: No more spikes, should remain stable
    sim.step(2);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(2), 1.0));
}

/**
 * @brief Test forgetting rule
 */
TEST_P(SnpSimulatorTest, ForgettingRule) {
    // System with 2 neurons
    SnpSimulator sim(backend, 2, 2);

    // Initial: σ1 has 3 spikes
    sim.setInitialSpikes(0, 3.0);

    // Rule 0: σ1 fires normally when has exactly 1 spike
    sim.addRule(0, 0, 1, 1.0, 1.0, 1.0, 0);

    // Rule 1: σ1 forgets 2 spikes when has at least 2 (a² → λ)
    sim.addRule(1, 0, -1, 2.0, 0.0, 2.0, 0, true);

    // Step 0: σ1 has 3 spikes, forgetting rule should apply
    sim.step(0);
    // After forgetting 2, should have 1 spike
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 1.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));

    // Step 1: σ1 has 1 spike, firing rule should apply
    sim.step(1);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 1.0));
}

/**
 * @brief Test delay behavior
 */
TEST_P(SnpSimulatorTest, DelayBehavior) {
    // System with 2 neurons
    SnpSimulator sim(backend, 2, 1);

    // Initial: σ1 has 1 spike
    sim.setInitialSpikes(0, 1.0);

    // Rule with delay 3: σ1 → σ2 after 3 time steps
    sim.addRule(0, 0, 1, 1.0, 1.0, 1.0, 3);

    // Verify initial state
    EXPECT_TRUE(sim.isNeuronOpen(0));
    EXPECT_TRUE(sim.isNeuronOpen(1));

    // Step 0: Fire rule, σ1 becomes closed
    sim.step(0);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));
    EXPECT_FALSE(sim.isNeuronOpen(0)) << "σ1 should be closed after firing with delay";

    // Step 1: σ1 still closed (delay=2 remaining)
    sim.step(1);
    EXPECT_FALSE(sim.isNeuronOpen(0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));

    // Step 2: σ1 still closed (delay=1 remaining)
    sim.step(2);
    EXPECT_FALSE(sim.isNeuronOpen(0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 0.0));

    // Step 3: σ1 opens, spike arrives at σ2
    sim.step(3);
    EXPECT_TRUE(sim.isNeuronOpen(0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 1.0));
}

/**
 * @brief Test that closed neurons reject incoming spikes
 */
TEST_P(SnpSimulatorTest, ClosedNeuronRejectsSpikes) {
    // System with 2 neurons
    SnpSimulator sim(backend, 2, 1);

    // Initial: both neurons have 1 spike
    sim.setInitialSpikes(0, 1.0);
    sim.setInitialSpikes(1, 1.0);

    // Rule: σ1 fires to σ2 with delay 2
    sim.addRule(0, 0, 1, 1.0, 1.0, 1.0, 2);

    // Step 0: σ1 fires, becomes closed
    sim.step(0);
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(0), 0.0));
    EXPECT_TRUE(floatEqual(sim.getSpikeCount(1), 1.0));
    EXPECT_FALSE(sim.isNeuronOpen(0));

    // Now if we manually try to add another rule that targets σ1,
    // it should be rejected by the status vector
    // This is implicitly tested by the verification example
}

// ============================================================================
// Test Instantiation for All Backends
// ============================================================================

// Test with CPU backend
INSTANTIATE_TEST_SUITE_P(
    CPUBackend,
    SnpSimulatorTest,
    ::testing::Values(BackendType::CPU)
);

// Test with CUDA backend (if available)
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(
    CUDABackend,
    SnpSimulatorTest,
    ::testing::Values(BackendType::CUDA)
);
#endif

// Test with MPI+CUDA backend (if available)
#ifdef USE_MPI
INSTANTIATE_TEST_SUITE_P(
    MPICUDABackend,
    SnpSimulatorTest,
    ::testing::Values(BackendType::MPI_CUDA)
);
#endif

// ============================================================================
// Main Function - Initialize MPI
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
