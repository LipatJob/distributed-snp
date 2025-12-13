# Makefile for distributed-snp project
# Convenience wrapper around CMake

BUILD_DIR := build
BUILD_TYPE ?= Release

# Compiler settings
CXX ?= g++
NVCC ?= nvcc
MPICC ?= mpicxx

# Build parallelization
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# CUDA architectures
CUDA_ARCH ?= 75

# Distribution settings
NODES ?= localhost 10.0.0.2
REMOTE_USER ?= $(USER)
REMOTE_DIR ?= ~/distributed-snp-new
HOSTFILE ?= hostfile.txt

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

.PHONY: all help configure build debug release clean rebuild install check-deps lint \
        generate-hostfile distribute check-nodes \
        deploy-tests test-distributed run-distributed \
        benchmark-sort benchmark-distributed compare-benchmarks visualize-benchmarks benchmark-and-visualize

# Default target
all: build

help:
	@echo "$(GREEN)Build Targets:$(NC)"
	@echo "  all               - Build the project (default)"
	@echo "  configure         - Configure CMake build"
	@echo "  build             - Build all targets"
	@echo "  debug             - Build with debug symbols"
	@echo "  release           - Build with optimizations"
	@echo "  clean             - Remove build directory"
	@echo "  rebuild           - Clean and build"
	@echo "  install           - Install libraries and headers"
	@echo "  check-deps        - Verify required dependencies"
	@echo "  lint              - Run linter on source files"
	@echo ""
	@echo "$(GREEN)Distribution Targets:$(NC)"
	@echo "  generate-hostfile - Generate MPI hostfile"
	@echo "  distribute        - Copy binaries to remote nodes"
	@echo "  check-nodes       - Check connectivity to remote nodes"
	@echo "  deploy-tests      - Deploy test executable to remote nodes"
	@echo "  test-distributed  - Deploy and run tests across nodes"
	@echo "  run-distributed   - Run demo on distributed nodes"
	@echo ""
	@echo "$(GREEN)Benchmark Targets:$(NC)"
	@echo "  benchmark-sort           - Run sort benchmarks locally"
	@echo "  benchmark-distributed    - Run sort benchmarks across distributed nodes"
	@echo "  compare-benchmarks       - Compare benchmark results (text)"
	@echo "  visualize-benchmarks     - Generate visualization plots from results"
	@echo "  benchmark-and-visualize  - Run benchmarks and auto-generate visualizations"
	@echo ""
	@echo "$(GREEN)Variables:$(NC)"
	@echo "  BUILD_TYPE        - Release or Debug (default: Release)"
	@echo "  CUDA_ARCH         - CUDA architectures (default: 75)"
	@echo "  JOBS              - Parallel jobs (default: auto-detected)"
	@echo "  CXX               - C++ compiler (default: g++)"
	@echo "  NVCC              - CUDA compiler (default: nvcc)"
	@echo "  MPICC             - MPI compiler (default: mpicxx)"
	@echo "  NODES             - Space-separated nodes (default: localhost 10.0.0.2)"
	@echo "  REMOTE_USER       - Username for remote nodes (default: current user)"
	@echo "  REMOTE_DIR        - Remote directory (default: ~/distributed-snp-new)"
	@echo "  HOSTFILE          - MPI hostfile path (default: hostfile.txt)"

configure:
	@echo "$(GREEN)Configuring CMake...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCH)"

build: configure
	@echo "$(GREEN)Building...$(NC)"
	@cmake --build $(BUILD_DIR) -j $(JOBS)
	@echo "$(GREEN)Build complete!$(NC)"

debug:
	@$(MAKE) build BUILD_TYPE=Debug

release:
	@$(MAKE) build BUILD_TYPE=Release

clean:
	@echo "$(YELLOW)Cleaning...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Done!$(NC)"

rebuild: clean all

install: build
	@echo "$(GREEN)Installing...$(NC)"
	@cmake --install $(BUILD_DIR)

check-deps:
	@echo "$(GREEN)Checking dependencies...$(NC)"
	@command -v cmake >/dev/null 2>&1 || { echo "$(YELLOW)cmake not found$(NC)"; exit 1; }
	@command -v $(CXX) >/dev/null 2>&1 || { echo "$(YELLOW)$(CXX) not found$(NC)"; exit 1; }
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "$(YELLOW)$(NVCC) not found$(NC)"; exit 1; }
	@command -v $(MPICC) >/dev/null 2>&1 || { echo "$(YELLOW)$(MPICC) not found$(NC)"; exit 1; }
	@echo "$(GREEN)All dependencies found!$(NC)"

lint:
	@echo "$(GREEN)Running linter...$(NC)"
	@command -v clang-tidy >/dev/null 2>&1 || { echo "$(YELLOW)clang-tidy not found. Install with: apt install clang-tidy$(NC)"; exit 1; }
	@find src tests -type f \( -name '*.cpp' -o -name '*.hpp' \) -print0 | \
		xargs -0 -P$(JOBS) -I{} sh -c 'echo "Linting {}..." && clang-tidy {} -- -std=c++17 -I./src' || true
	@echo "$(GREEN)Linting complete!$(NC)"

generate-hostfile:
	@echo "$(GREEN)Generating hostfile...$(NC)"
	@rm -f $(HOSTFILE)
	@for node in $(NODES); do \
		echo "$$node slots=1" >> $(HOSTFILE); \
	done
	@echo "$(GREEN)Generated: $(HOSTFILE)$(NC)"
	@cat $(HOSTFILE)

distribute: build
	@echo "$(GREEN)Distributing binaries...$(NC)"
	@for node in $(NODES); do \
		echo "$(BLUE)Copying to $$node...$(NC)"; \
		scp -r $(BUILD_DIR)/bin $(REMOTE_USER)@$$node:$(REMOTE_DIR)/; \
		scp -r $(BUILD_DIR)/lib $(REMOTE_USER)@$$node:$(REMOTE_DIR)/; \
		echo "$(GREEN)✓ $$node$(NC)"; \
	done
	@echo "$(GREEN)Distribution complete!$(NC)"

check-nodes:
	@echo "$(GREEN)Checking connectivity...$(NC)"
	@for node in $(NODES); do \
		echo -n "  $$node: "; \
		ssh -o ConnectTimeout=5 $(REMOTE_USER)@$$node "echo '$(GREEN)✓$(NC)'" 2>/dev/null || echo "$(YELLOW)✗$(NC)"; \
	done

deploy-tests: build
	@echo "$(GREEN)Deploying tests...$(NC)"
	@chmod +x scripts/deploy_tests.sh
	@./scripts/deploy_tests.sh
	@echo "$(GREEN)Done!$(NC)"

test-distributed: deploy-tests
	@echo "$(GREEN)Running distributed tests...$(NC)"
	@chmod +x scripts/run_distributed_tests.sh
	@./scripts/run_distributed_tests.sh

run-distributed: distribute generate-hostfile
	@echo "$(GREEN)Running on distributed nodes...$(NC)"
	@mpirun --hostfile $(HOSTFILE) \
		--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
		$(REMOTE_DIR)/bin/matrix_demo

# ============================================================================
# Benchmark Targets
# ============================================================================

benchmark-sort: build
	@echo "$(GREEN)Running sort benchmarks locally...$(NC)"
	@chmod +x scripts/run_benchmark.sh
	@./scripts/run_benchmark.sh

benchmark-distributed: build
	@echo "$(GREEN)Running distributed sort benchmarks...$(NC)"
	@chmod +x scripts/run_distributed_benchmark.sh
	@./scripts/run_distributed_benchmark.sh $(BENCHMARK_ARGS)

compare-benchmarks:
	@echo "$(GREEN)Comparing benchmark results...$(NC)"
	@chmod +x scripts/compare_benchmarks.py
	@./scripts/compare_benchmarks.py benchmark_results/*.json || echo "$(YELLOW)No benchmark results found. Run benchmarks first.$(NC)"

visualize-benchmarks:
	@echo "$(GREEN)Visualizing benchmark results...$(NC)"
	@chmod +x scripts/visualize_benchmarks.py
	@mkdir -p benchmark_results
	@./scripts/visualize_benchmarks.py benchmark_results/*.json || echo "$(YELLOW)No benchmark results found. Run benchmarks first.$(NC)"

benchmark-and-visualize: build
	@echo "$(GREEN)Running benchmarks with automatic visualization...$(NC)"
	@chmod +x scripts/benchmark_and_visualize.sh
	@./scripts/benchmark_and_visualize.sh
