# Makefile for new-snp project
# This is a convenience wrapper around CMake

# Build directory
BUILD_DIR := build
BUILD_TYPE ?= Release

# Compiler settings
CXX ?= g++
NVCC ?= nvcc
MPICC ?= mpicxx

# Number of parallel jobs
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# CUDA architectures (adjust based on your GPU)
CUDA_ARCH ?= 75

# Distribution settings
NODES ?= localhost 10.0.0.2
REMOTE_USER ?= $(USER)
REMOTE_DIR ?= ~/distributed-snp-new
HOSTFILE ?= hostfile.txt

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: all build clean test test-distributed deploy-tests install configure help debug release run distribute generate-hostfile run-distributed snp-demo test-snp

# Default target
all: build

# Display help
help:
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  all          - Build the project (default)"
	@echo "  configure    - Configure CMake build"
	@echo "  build        - Build all targets"
	@echo "  debug        - Build with debug symbols"
	@echo "  release      - Build with optimizations (default)"
	@echo "  run          - Run the matrix demo locally"
	@echo "  snp-demo     - Run the SNP simulator demo locally"
	@echo "  test         - Run tests locally"
	@echo "  test-snp     - Run SNP simulator tests locally"
	@echo "  deploy-tests - Deploy test executable to remote nodes"
	@echo "  test-distributed - Deploy and run tests across distributed nodes"
	@echo "  distribute   - Distribute binary to remote nodes"
	@echo "  generate-hostfile - Generate MPI hostfile"
	@echo "  run-distributed - Run demo on distributed nodes"
	@echo "  clean        - Remove build directory"
	@echo "  install      - Install libraries and headers"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "$(GREEN)Environment variables:$(NC)"
	@echo "  BUILD_TYPE   - Release or Debug (default: Release)"
	@echo "  CUDA_ARCH    - CUDA architectures (default: 75,80,86)"
	@echo "  JOBS         - Number of parallel jobs (default: auto-detected)"
	@echo "  CXX          - C++ compiler (default: g++)"
	@echo "  NVCC         - CUDA compiler (default: nvcc)"
	@echo "  MPICC        - MPI C++ compiler (default: mpicxx)"
	@echo ""
	@echo "$(GREEN)Distribution variables:$(NC)"
	@echo "  NODES        - Space-separated list of nodes (default: node1 node2 node3 node4)"
	@echo "  REMOTE_USER  - Username for remote nodes (default: current user)"
	@echo "  REMOTE_DIR   - Remote directory path (default: ~/new-snp)"
	@echo "  HOSTFILE     - MPI hostfile path (default: hostfile.txt)"

# Configure the build
configure:
	@echo "$(GREEN)Configuring CMake...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCH)"

# Build all targets
build: configure
	@echo "$(GREEN)Building project...$(NC)"
	@cmake --build $(BUILD_DIR) -j $(JOBS)
	@echo "$(GREEN)Build complete!$(NC)"

# Build with debug flags
debug:
	@$(MAKE) build BUILD_TYPE=Debug

# Build with release flags
release:
	@$(MAKE) build BUILD_TYPE=Release

# Deploy tests to distributed nodes
deploy-tests: build
	@echo "$(GREEN)Deploying tests to distributed nodes...$(NC)"
	@chmod +x scripts/deploy_tests.sh
	@./scripts/deploy_tests.sh
	@echo "$(GREEN)Test deployment complete!$(NC)"

# Run distributed tests
test-distributed: deploy-tests
	@echo "$(GREEN)Running distributed tests...$(NC)"
	@chmod +x scripts/run_distributed_tests.sh
	@./scripts/run_distributed_tests.sh

# Install
install: build
	@echo "$(GREEN)Installing...$(NC)"
	@cmake --install $(BUILD_DIR)

# Clean build directory
clean:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Clean complete!$(NC)"

# Quick rebuild
rebuild: clean all

# Check dependencies
check-deps:
	@echo "$(GREEN)Checking dependencies...$(NC)"
	@command -v cmake >/dev/null 2>&1 || { echo "$(YELLOW)cmake is not installed$(NC)"; exit 1; }
	@command -v $(CXX) >/dev/null 2>&1 || { echo "$(YELLOW)C++ compiler $(CXX) not found$(NC)"; exit 1; }
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "$(YELLOW)CUDA compiler $(NVCC) not found$(NC)"; exit 1; }
	@command -v $(MPICC) >/dev/null 2>&1 || { echo "$(YELLOW)MPI compiler $(MPICC) not found$(NC)"; exit 1; }
	@echo "$(GREEN)All dependencies found!$(NC)"

# Generate MPI hostfile
generate-hostfile:
	@echo "$(GREEN)Generating hostfile...$(NC)"
	@rm -f $(HOSTFILE)
	@for node in $(NODES); do \
		echo "$$node slots=1" >> $(HOSTFILE); \
	done
	@echo "$(GREEN)Hostfile generated: $(HOSTFILE)$(NC)"
	@echo "$(BLUE)Contents:$(NC)"
	@cat $(HOSTFILE)

# Distribute binary and libraries to remote nodes
distribute: build
	@echo "$(GREEN)Distributing binaries to remote nodes...$(NC)"
	@for node in $(NODES); do \
		echo "$(BLUE)Copying to $$node...$(NC)"; \
		scp -r $(BUILD_DIR)/bin $(REMOTE_USER)@$$node:$(REMOTE_DIR)/; \
		scp -r $(BUILD_DIR)/lib $(REMOTE_USER)@$$node:$(REMOTE_DIR)/; \
		echo "$(GREEN)✓ Completed $$node$(NC)"; \
	done
	@echo "$(GREEN)Distribution complete!$(NC)"

# Run on distributed nodes
run-distributed: distribute generate-hostfile
	@echo "$(GREEN)Running matrix demo on distributed nodes...$(NC)"
	@mpirun --hostfile $(HOSTFILE) \
		--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
	 $(REMOTE_DIR)/bin/matrix_demo

# Run SNP demo locally
snp-demo: build
	@echo "$(GREEN)Running SNP simulator demo...$(NC)"
	@mpirun -np 1 $(BUILD_DIR)/snp_demo --allow-run-as-root

# Run SNP tests locally
test-snp: build
	@echo "$(GREEN)Running SNP simulator tests...$(NC)"
	@mpirun -np 2 $(BUILD_DIR)/test_snp_simulator --allow-run-as-root --oversubscribe

# Check connectivity to all nodes
check-nodes:
	@echo "$(GREEN)Checking connectivity to nodes...$(NC)"
	@for node in $(NODES); do \
		echo -n "  $$node: "; \
		ssh -o ConnectTimeout=5 $(REMOTE_USER)@$$node "echo '$(GREEN)✓ reachable$(NC)'" 2>/dev/null || echo "$(YELLOW)✗ unreachable$(NC)"; \
	done
