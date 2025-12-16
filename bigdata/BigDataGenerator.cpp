#include "BigDataCommon.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <map>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;
using namespace bigdata;

struct Config {
    uint64_t total_neurons = 1000000;
    int num_ranks = 2;
    double avg_intra_degree = 10.0;
    double avg_inter_degree = 1.0;
    std::string out_dir = "output/bigdata";
    uint64_t seed = 123;
    double mem_limit_gb = 0.0; // 0 means no check
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -n, --neurons N       Total number of neurons (default: 1000000)\n"
              << "  -r, --ranks R         Number of ranks/nodes (default: 2)\n"
              << "  --intra D             Average intra-node degree (default: 10.0)\n"
              << "  --inter D             Average inter-node degree (default: 1.0)\n"
              << "  -o, --outdir DIR      Output directory (default: output/bigdata)\n"
              << "  -s, --seed S          Random seed (default: 123)\n"
              << "  --mem-limit GB        Memory limit check in GB (default: 0/off)\n"
              << "  -h, --help            Show this help\n";
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--neurons") c.total_neurons = std::stoull(argv[++i]);
        else if (arg == "-r" || arg == "--ranks") c.num_ranks = std::stoi(argv[++i]);
        else if (arg == "--intra") c.avg_intra_degree = std::stod(argv[++i]);
        else if (arg == "--inter") c.avg_inter_degree = std::stod(argv[++i]);
        else if (arg == "-o" || arg == "--outdir") c.out_dir = argv[++i];
        else if (arg == "-s" || arg == "--seed") c.seed = std::stoull(argv[++i]);
        else if (arg == "--mem-limit") c.mem_limit_gb = std::stod(argv[++i]);
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); exit(0); }
    }
    return c;
}

// Helper to write binary data
template<typename T>
void write_bin(std::ofstream& out, const T& data) {
    out.write(reinterpret_cast<const char*>(&data), sizeof(T));
}

template<typename T>
void write_vec(std::ofstream& out, const std::vector<T>& vec) {
    out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);

    // 1. Memory Check
    // Estimate size:
    // Neurons: ~64 bytes (struct + vector overhead) -> Let's say 100 bytes conservative
    // Synapses: 12 bytes
    // Total = N * 100 + N * (intra + inter) * 12
    double estimated_size_bytes = config.total_neurons * 100.0 + 
                                  config.total_neurons * (config.avg_intra_degree + config.avg_inter_degree) * 12.0;
    double estimated_size_gb = estimated_size_bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Estimated Dataset Size: " << estimated_size_gb << " GB" << std::endl;

    if (config.mem_limit_gb > 0) {
        if (estimated_size_gb < config.mem_limit_gb) {
            std::cerr << "Error: Dataset too small! Estimated " << estimated_size_gb 
                      << " GB < Limit " << config.mem_limit_gb << " GB" << std::endl;
            return 1;
        } else {
            std::cout << "Memory check passed: Dataset is larger than " << config.mem_limit_gb << " GB" << std::endl;
        }
    }

    // 2. Setup
    fs::create_directories(config.out_dir);
    fs::path temp_dir = fs::path(config.out_dir) / "temp";
    fs::create_directories(temp_dir);

    std::mt19937_64 rng(config.seed);
    
    // Distribution
    uint64_t neurons_per_rank = config.total_neurons / config.num_ranks;
    uint64_t remainder = config.total_neurons % config.num_ranks;

    auto get_neuron_rank = [&](uint64_t id) -> int {
        // Simple block distribution
        // Rank 0 gets neurons_per_rank + (remainder > 0 ? 1 : 0)
        // But to keep it simple, let's just calculate ranges
        // Actually, we iterate rank by rank, so we know the ranges.
        // We need reverse mapping for destination.
        // Let's precalculate start indices.
        return std::min((uint64_t)config.num_ranks - 1, id / (neurons_per_rank + (remainder > 0 ? 1 : 0))); // Approx
        // Better:
        // If remainder=0, id / neurons_per_rank.
        // If remainder!=0, it's trickier.
        // Let's stick to: First 'remainder' ranks get +1 neuron.
        uint64_t large_ranks_count = remainder;
        uint64_t large_rank_size = neurons_per_rank + 1;
        uint64_t small_rank_start_id = large_ranks_count * large_rank_size;
        
        if (id < small_rank_start_id) return id / large_rank_size;
        return large_ranks_count + (id - small_rank_start_id) / neurons_per_rank;
    };

    auto get_rank_start_id = [&](int r) -> uint64_t {
        uint64_t large_ranks_count = remainder;
        uint64_t large_rank_size = neurons_per_rank + 1;
        if ((uint64_t)r < large_ranks_count) return r * large_rank_size;
        return large_ranks_count * large_rank_size + (r - large_ranks_count) * neurons_per_rank;
    };

    auto get_rank_size = [&](int r) -> uint64_t {
        return neurons_per_rank + ((uint64_t)r < remainder ? 1 : 0);
    };

    // Track export counts for each pair (src_rank, dst_rank)
    // We need this to assign export_index.
    // Since we process src_rank sequentially, we can just keep a vector of counters for current src_rank.
    // But we need to write to temp files.

    // 3. Generate
    std::cout << "Generating partitions..." << std::endl;

    for (int src_rank = 0; src_rank < config.num_ranks; ++src_rank) {
        uint64_t start_id = get_rank_start_id(src_rank);
        uint64_t count = get_rank_size(src_rank);
        
        std::cout << "  Processing Rank " << src_rank << " (" << count << " neurons)..." << std::endl;

        // Open temp files
        std::ofstream f_neurons(temp_dir / ("neurons_" + std::to_string(src_rank)), std::ios::binary);
        std::ofstream f_local_syn(temp_dir / ("local_syn_" + std::to_string(src_rank)), std::ios::binary);
        
        // Map dst_rank -> ofstream
        std::map<int, std::ofstream> f_exports;
        std::map<int, std::ofstream> f_imports; // We write to dst_rank's import file
        std::map<int, int> export_counters; // dst_rank -> count

        // Pre-open export/import files
        for (int dst_rank = 0; dst_rank < config.num_ranks; ++dst_rank) {
            if (src_rank == dst_rank) continue;
            
            // Export file: src_rank sends to dst_rank
            f_exports[dst_rank].open(temp_dir / ("export_" + std::to_string(src_rank) + "_" + std::to_string(dst_rank)), std::ios::binary);
            
            // Import file: dst_rank receives from src_rank
            // We append to it? No, multiple src_ranks write to same dst_rank's import file?
            // No, we create unique file per pair: import_dst_src
            f_imports[dst_rank].open(temp_dir / ("import_" + std::to_string(dst_rank) + "_" + std::to_string(src_rank)), std::ios::binary);
            
            export_counters[dst_rank] = 0;
        }

        std::poisson_distribution<> dist_intra(config.avg_intra_degree);
        std::poisson_distribution<> dist_inter(config.avg_inter_degree);
        std::uniform_int_distribution<int> dist_weight(1, 10);
        std::uniform_int_distribution<int> dist_spikes(0, 5);

        uint64_t local_syn_count = 0;

        for (uint64_t i = 0; i < count; ++i) {
            uint64_t global_id = start_id + i;
            
            // Write Neuron
            int32_t id = (int32_t)global_id;
            int32_t init_spikes = dist_spikes(rng);
            int32_t num_rules = 1;
            write_bin(f_neurons, id);
            write_bin(f_neurons, init_spikes);
            write_bin(f_neurons, num_rules);
            
            // Default Rule: a -> a
            RuleData rule{1, 1, 1, 0};
            write_bin(f_neurons, rule);

            // Generate Intra-edges
            int n_intra = dist_intra(rng);
            for (int k = 0; k < n_intra; ++k) {
                // Pick random local dest
                std::uniform_int_distribution<uint64_t> dist_local_dst(0, count - 1);
                uint64_t dst_local_idx = dist_local_dst(rng);
                
                LocalSynapseData syn;
                syn.source_local_idx = (int32_t)i;
                syn.dest_local_idx = (int32_t)dst_local_idx;
                syn.weight = dist_weight(rng);
                write_bin(f_local_syn, syn);
                local_syn_count++;
            }

            // Generate Inter-edges
            int n_inter = dist_inter(rng);
            for (int k = 0; k < n_inter; ++k) {
                // Pick random remote rank
                std::uniform_int_distribution<int> dist_rank(0, config.num_ranks - 1);
                int dst_rank = dist_rank(rng);
                if (dst_rank == src_rank) {
                    // Fallback to local or retry? Let's just make it local
                    // Or retry once
                    dst_rank = dist_rank(rng);
                    if (dst_rank == src_rank) continue; // Skip
                }

                // Pick random neuron in dst_rank
                uint64_t dst_count = get_rank_size(dst_rank);
                std::uniform_int_distribution<uint64_t> dist_remote_dst(0, dst_count - 1);
                uint64_t dst_local_idx = dist_remote_dst(rng);

                int export_idx = export_counters[dst_rank]++;
                
                ExportSynapseData ex_syn;
                ex_syn.source_local_idx = (int32_t)i;
                ex_syn.weight = dist_weight(rng);
                write_bin(f_exports[dst_rank], ex_syn);

                ImportSynapseData im_syn;
                im_syn.export_index = export_idx;
                im_syn.dest_local_idx = (int32_t)dst_local_idx;
                write_bin(f_imports[dst_rank], im_syn);
            }
        }
        
        // Store metadata for assembly
        // We need to know how many local synapses, etc.
        // We can just read file size later.
    }

    // 4. Assemble Partitions
    std::cout << "Assembling partitions..." << std::endl;
    
    std::ofstream desc_file(fs::path(config.out_dir) / "descriptor.json");
    desc_file << "{\n";
    desc_file << "  \"global_neurons\": " << config.total_neurons << ",\n";
    desc_file << "  \"nodes\": " << config.num_ranks << ",\n";
    desc_file << "  \"partitions\": [\n";

    for (int r = 0; r < config.num_ranks; ++r) {
        std::string part_filename = "partition_" + std::to_string(r) + ".dat";
        fs::path part_path = fs::path(config.out_dir) / part_filename;
        std::ofstream out(part_path, std::ios::binary);

        // Calculate counts
        uint64_t num_local_neurons = get_rank_size(r);
        uint64_t num_local_synapses = fs::file_size(temp_dir / ("local_syn_" + std::to_string(r))) / sizeof(LocalSynapseData);
        
        uint64_t num_export_groups = 0;
        for (int dst = 0; dst < config.num_ranks; ++dst) {
            if (r == dst) continue;
            if (fs::exists(temp_dir / ("export_" + std::to_string(r) + "_" + std::to_string(dst)))) {
                if (fs::file_size(temp_dir / ("export_" + std::to_string(r) + "_" + std::to_string(dst))) > 0)
                    num_export_groups++;
            }
        }

        uint64_t num_import_groups = 0;
        for (int src = 0; src < config.num_ranks; ++src) {
            if (r == src) continue;
            if (fs::exists(temp_dir / ("import_" + std::to_string(r) + "_" + std::to_string(src)))) {
                if (fs::file_size(temp_dir / ("import_" + std::to_string(r) + "_" + std::to_string(src))) > 0)
                    num_import_groups++;
            }
        }

        // Write Header
        PartitionHeader header;
        header.magic = PARTITION_MAGIC;
        header.rank_id = r;
        header.num_ranks = config.num_ranks;
        header.num_local_neurons = num_local_neurons;
        header.num_local_synapses = num_local_synapses;
        header.num_export_groups = num_export_groups;
        header.num_import_groups = num_import_groups;
        write_bin(out, header);

        // Copy Neurons
        {
            std::ifstream in(temp_dir / ("neurons_" + std::to_string(r)), std::ios::binary);
            out << in.rdbuf();
        }

        // Copy Local Synapses
        {
            std::ifstream in(temp_dir / ("local_syn_" + std::to_string(r)), std::ios::binary);
            out << in.rdbuf();
        }

        // Write Export Groups
        for (int dst = 0; dst < config.num_ranks; ++dst) {
            if (r == dst) continue;
            fs::path p = temp_dir / ("export_" + std::to_string(r) + "_" + std::to_string(dst));
            if (fs::exists(p) && fs::file_size(p) > 0) {
                ExportGroupHeader gh;
                gh.target_rank = dst;
                gh.num_synapses = fs::file_size(p) / sizeof(ExportSynapseData);
                write_bin(out, gh);
                
                std::ifstream in(p, std::ios::binary);
                out << in.rdbuf();
            }
        }

        // Write Import Groups
        for (int src = 0; src < config.num_ranks; ++src) {
            if (r == src) continue;
            fs::path p = temp_dir / ("import_" + std::to_string(r) + "_" + std::to_string(src));
            if (fs::exists(p) && fs::file_size(p) > 0) {
                ImportGroupHeader gh;
                gh.source_rank = src;
                gh.num_synapses = fs::file_size(p) / sizeof(ImportSynapseData);
                write_bin(out, gh);
                
                std::ifstream in(p, std::ios::binary);
                out << in.rdbuf();
            }
        }

        desc_file << "    \"" << part_filename << "\"" << (r == config.num_ranks - 1 ? "" : ",") << "\n";
    }

    desc_file << "  ]\n";
    desc_file << "}\n";

    // Cleanup
    fs::remove_all(temp_dir);

    std::cout << "Done. Output in " << config.out_dir << std::endl;
    return 0;
}
