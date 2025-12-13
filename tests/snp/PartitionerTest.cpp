#include <gtest/gtest.h>
#include "SnpSystemConfig.hpp"
#include "LinearPartitioner.hpp"
#include "RedBluePartitioner.hpp"
#include "LouvainPartitioner.hpp"
#include <vector>
#include <memory>
#include <algorithm>

// Helper to create a simple line graph: 0-1-2-3-...-(N-1)
SnpSystemConfig createLineGraph(int num_neurons) {
    SnpSystemConfig config;
    for (int i = 0; i < num_neurons; ++i) {
        config.neurons.emplace_back(i, 0);
    }
    for (int i = 0; i < num_neurons - 1; ++i) {
        config.synapses.emplace_back(i, i + 1);
    }
    return config;
}

// Helper to create two disconnected components
// Component 1: 0-1-2
// Component 2: 3-4-5
SnpSystemConfig createDisconnectedGraph() {
    SnpSystemConfig config;
    for (int i = 0; i < 6; ++i) {
        config.neurons.emplace_back(i, 0);
    }
    // Component 1
    config.synapses.emplace_back(0, 1);
    config.synapses.emplace_back(1, 2);
    config.synapses.emplace_back(2, 0); // Cycle
    
    // Component 2
    config.synapses.emplace_back(3, 4);
    config.synapses.emplace_back(4, 5);
    config.synapses.emplace_back(5, 3); // Cycle
    return config;
}

TEST(PartitionerTest, LinearPartitioner_EvenSplit) {
    LinearPartitioner partitioner;
    int num_neurons = 10;
    int num_partitions = 2;
    auto config = createLineGraph(num_neurons);
    
    auto assignments = partitioner.partition(config, num_partitions);
    
    ASSERT_EQ(assignments.size(), num_neurons);
    
    // Expect 0-4 to be in partition 0, 5-9 in partition 1
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(assignments[i], 0) << "Neuron " << i << " should be in partition 0";
    }
    for (int i = 5; i < 10; ++i) {
        EXPECT_EQ(assignments[i], 1) << "Neuron " << i << " should be in partition 1";
    }
}

TEST(PartitionerTest, LinearPartitioner_UnevenSplit) {
    LinearPartitioner partitioner;
    int num_neurons = 10;
    int num_partitions = 3;
    auto config = createLineGraph(num_neurons);
    
    auto assignments = partitioner.partition(config, num_partitions);
    
    // Chunk size = ceil(10/3) = 4
    // P0: 0,1,2,3
    // P1: 4,5,6,7
    // P2: 8,9
    
    for (int i = 0; i < 4; ++i) EXPECT_EQ(assignments[i], 0);
    for (int i = 4; i < 8; ++i) EXPECT_EQ(assignments[i], 1);
    for (int i = 8; i < 10; ++i) EXPECT_EQ(assignments[i], 2);
}

TEST(PartitionerTest, RedBluePartitioner_LineGraph) {
    RedBluePartitioner partitioner;
    int num_neurons = 10;
    int num_partitions = 2;
    auto config = createLineGraph(num_neurons);
    
    auto assignments = partitioner.partition(config, num_partitions);
    
    // BFS should fill partition 0 with connected nodes starting from seed (likely 0)
    // It should fill roughly half.
    // Since it's a line 0-1-2-3-4-5-6-7-8-9, BFS from 0 will pick 0,1,2,3,4...
    
    int count0 = 0;
    int count1 = 0;
    for (int p : assignments) {
        if (p == 0) count0++;
        else if (p == 1) count1++;
    }
    
    EXPECT_EQ(count0 + count1, num_neurons);
    EXPECT_NEAR(count0, 5, 1); // Should be roughly balanced
    EXPECT_NEAR(count1, 5, 1);
    
    // Verify contiguity (heuristic check for line graph BFS)
    // We expect a split point.
    int changes = 0;
    for (size_t i = 0; i < assignments.size() - 1; ++i) {
        if (assignments[i] != assignments[i+1]) changes++;
    }
    EXPECT_LE(changes, 1) << "Line graph should have at most 1 cut with BFS partitioning";
}

TEST(PartitionerTest, RedBluePartitioner_Disconnected) {
    RedBluePartitioner partitioner;
    int num_partitions = 2;
    auto config = createDisconnectedGraph(); // 6 nodes, 2 components of 3
    
    auto assignments = partitioner.partition(config, num_partitions);
    
    // Ideally, one component goes to P0, other to P1
    // Or BFS fills P0 with one component (size 3), then moves to next.
    // Target size = 6/2 = 3.
    
    // Check that nodes 0,1,2 are in same partition
    EXPECT_EQ(assignments[0], assignments[1]);
    EXPECT_EQ(assignments[1], assignments[2]);
    
    // Check that nodes 3,4,5 are in same partition
    EXPECT_EQ(assignments[3], assignments[4]);
    EXPECT_EQ(assignments[4], assignments[5]);
    
    // Check balance
    int count0 = 0;
    for(int p : assignments) if(p == 0) count0++;
    EXPECT_EQ(count0, 3);
}

TEST(PartitionerTest, LouvainPartitioner_Disconnected) {
    LouvainPartitioner partitioner;
    int num_partitions = 2;
    auto config = createDisconnectedGraph();
    
    auto assignments = partitioner.partition(config, num_partitions);
    
    // Louvain should easily identify the two cliques/cycles as communities
    // and assign them to different partitions if balancing is working.
    
    // Check that nodes 0,1,2 are in same partition
    EXPECT_EQ(assignments[0], assignments[1]);
    EXPECT_EQ(assignments[1], assignments[2]);
    
    // Check that nodes 3,4,5 are in same partition
    EXPECT_EQ(assignments[3], assignments[4]);
    EXPECT_EQ(assignments[4], assignments[5]);
    
    // Since they are disconnected, they should likely be in different partitions 
    // to balance load (3 vs 3).
    EXPECT_NE(assignments[0], assignments[3]);
}

TEST(PartitionerTest, GetTypeAndName) {
    LinearPartitioner linear;
    EXPECT_EQ(linear.getType(), PartitionerType::LINEAR);
    EXPECT_EQ(IPartitioner::getPartitionerName(linear.getType()), "Linear (Block) Partitioning");

    RedBluePartitioner rb;
    EXPECT_EQ(rb.getType(), PartitionerType::RED_BLUE_BFS);
    EXPECT_EQ(IPartitioner::getPartitionerName(rb.getType()), "Red-Blue Pebbling (BFS)");

    LouvainPartitioner louvain;
    EXPECT_EQ(louvain.getType(), PartitionerType::LOUVAIN);
    EXPECT_EQ(IPartitioner::getPartitionerName(louvain.getType()), "Louvain Community Detection");
}
