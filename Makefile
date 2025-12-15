# Makefile for distributed-snp project
# Optimized for Caching, Parallelism, and HPC Deployment

# --- Configuration ---
BUILD_DIR   := build
BUILD_TYPE  ?= Release
CXX         ?= g++
NVCC        ?= nvcc
MPICC       ?= mpicxx
CUDA_ARCH   ?= 75

# Auto-detect parallelism
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Deployment Settings
NODES       ?= localhost 10.0.0.2
REMOTE_USER ?= $(USER)
REMOTE_DIR  ?= /home/shared/tmp/distributed-snp-new
HOSTFILE    ?= hostfile.txt

# --- Caching & Optimization Tools ---
# Auto-detect ccache for faster recompilation
CCACHE_EXE := $(shell command -v ccache 2> /dev/null)
ifdef CCACHE_EXE
    CMAKE_CCACHE_OPT := -DCMAKE_CXX_COMPILER_LAUNCHER=$(CCACHE_EXE) -DCMAKE_CUDA_COMPILER_LAUNCHER=$(CCACHE_EXE)
endif

# Auto-detect Ninja for faster build generation
NINJA_EXE := $(shell command -v ninja 2> /dev/null)
ifdef NINJA_EXE
    CMAKE_GEN_OPT := -G Ninja
else
    CMAKE_GEN_OPT :=
endif

# --- Colors ---
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
NC     := \033[0m

# --- Main Targets ---
.PHONY: all help build clean rebuild install distclean \
        lint format compile-commands \
        generate-hostfile distribute check-nodes \
        test benchmark benchmark-viz profile

all: build

# Dynamic Help Generation
help: ## Show this help message
	@echo "$(BLUE)Distributed SNP Build System$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2}'

# --- Build System ---

# Only run cmake if cache doesn't exist or CMakeLists changed
$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	@echo "$(BLUE)Configuring CMake ($(BUILD_TYPE))...$(NC)"
	@cmake -B $(BUILD_DIR) -S . \
		$(CMAKE_GEN_OPT) \
		$(CMAKE_CCACHE_OPT) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)

build: $(BUILD_DIR)/CMakeCache.txt ## Build the project (incremental)
	@echo "$(GREEN)Building project...$(NC)"
	@echo "Using ccache: $(if $(CCACHE_EXE),yes,no). Using ninja : $(if $(NINJA_EXE),yes,no)"
	@cmake --build $(BUILD_DIR) -j $(JOBS)

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@cmake --build $(BUILD_DIR) --target clean 2>/dev/null || rm -rf $(BUILD_DIR)

compile-commands: $(BUILD_DIR)/CMakeCache.txt ## Link compile_commands.json for LSP support
	@ln -sf $(BUILD_DIR)/compile_commands.json .

# --- Distribution & MPI ---

generate-hostfile: ## Generate MPI hostfile from NODES variable
	@echo "$(GREEN)Generating hostfile...$(NC)"
	@rm -f $(HOSTFILE)
	@for node in $(NODES); do echo "$$node slots=1" >> $(HOSTFILE); done
	@cat $(HOSTFILE)

distribute: build ## Deploy binaries to nodes using rsync (Fast)
	@echo "$(GREEN)Distributing binaries (using rsync)...$(NC)"
	@for node in $(NODES); do \
		echo "$(BLUE)Syncing to $$node...$(NC)"; \
		ssh $(REMOTE_USER)@$$node "mkdir -p $(REMOTE_DIR)"; \
		rsync -azP --delete $(BUILD_DIR)/bin $(BUILD_DIR)/lib $(REMOTE_USER)@$$node:$(REMOTE_DIR)/; \
	done
	@echo "$(GREEN)Distribution complete!$(NC)"

check-nodes: ## Verify SSH connectivity to nodes
	@echo "$(GREEN)Checking node connectivity...$(NC)"
	@for node in $(NODES); do \
		printf "  %-15s " "$$node:"; \
		ssh -o ConnectTimeout=3 $(REMOTE_USER)@$$node "echo 'OK'" >/dev/null 2>&1 \
		&& echo "$(GREEN)✓$(NC)" || echo "$(YELLOW)✗$(NC)"; \
	done

# --- Execution Wrappers ---

test: distribute ## Run distributed tests
	@echo "$(GREEN)Running distributed tests...$(NC)"
	@./scripts/run_tests.sh $(ARGS)

benchmark: distribute ## Run benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	@./scripts/run_benchmark.sh $(ARGS)

benchmark-viz: build ## Run benchmarks and visualize
	@echo "$(GREEN)Running benchmarks + visualization...$(NC)"
	@./scripts/benchmark_and_visualize.sh

profile: distribute ## Profile implementations
	@echo "$(GREEN)Profiling...$(NC)"
	@./scripts/run_profiling.sh $(ARGS)