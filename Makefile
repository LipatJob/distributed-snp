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
NODES ?= node1 node2 node3 node4
REMOTE_USER ?= $(USER)
REMOTE_DIR ?= ~/new-snp
HOSTFILE ?= hostfile.txt

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: all build clean test install configure help debug release run distribute generate-hostfile run-distributed

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
	@echo "  test         - Run tests"
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

# Run the demo
run: build
	@echo "$(GREEN)Running matrix demo...$(NC)"
	@mpirun -np 4 $(BUILD_DIR)/matrix_demo

# Run tests
test: build
	@echo "$(GREEN)Running tests...$(NC)"
	@cd $(BUILD_DIR) && ctest --output-on-failure

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
		echo "$$node slots=4" >> $(HOSTFILE); \
	done
	@echo "$(GREEN)Hostfile generated: $(HOSTFILE)$(NC)"
	@echo "$(BLUE)Contents:$(NC)"
	@cat $(HOSTFILE)

# Distribute binary and libraries to remote nodes
distribute: build
	@echo "$(GREEN)Distributing binaries to remote nodes...$(NC)"
	@for node in $(NODES); do \
		echo "$(BLUE)Copying to $$node...$(NC)"; \
		ssh $(REMOTE_USER)@$$node "mkdir -p $(REMOTE_DIR)/bin $(REMOTE_DIR)/lib" || exit 1; \
		rsync -avz --progress $(BUILD_DIR)/matrix_demo $(REMOTE_USER)@$$node:$(REMOTE_DIR)/bin/ || exit 1; \
		rsync -avz --progress $(BUILD_DIR)/test_matrix_ops $(REMOTE_USER)@$$node:$(REMOTE_DIR)/bin/ 2>/dev/null || true; \
		rsync -avz --progress $(BUILD_DIR)/*.so $(BUILD_DIR)/*.a $(REMOTE_USER)@$$node:$(REMOTE_DIR)/lib/ 2>/dev/null || true; \
		echo "$(GREEN)✓ Completed $$node$(NC)"; \
	done
	@echo "$(GREEN)Distribution complete!$(NC)"

# Run on distributed nodes
run-distributed: distribute generate-hostfile
	@echo "$(GREEN)Running matrix demo on distributed nodes...$(NC)"
	@mpirun --hostfile $(HOSTFILE) $(REMOTE_DIR)/bin/matrix_demo

# Check connectivity to all nodes
check-nodes:
	@echo "$(GREEN)Checking connectivity to nodes...$(NC)"
	@for node in $(NODES); do \
		echo -n "  $$node: "; \
		ssh -o ConnectTimeout=5 $(REMOTE_USER)@$$node "echo '$(GREEN)✓ reachable$(NC)'" 2>/dev/null || echo "$(YELLOW)✗ unreachable$(NC)"; \
	done
