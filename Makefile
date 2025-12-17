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
NODES       ?= localhost 10.0.0.2 10.0.1.2
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
BLUE   := \033[0;34m
NC     := \033[0m

# --- Main Targets ---
.PHONY: all help build clean rebuild install distclean \
        lint format compile-commands \
        generate-hostfile distribute check-nodes \
        test benchmark benchmark-viz profile

all: distribute
	@./scripts/run_all.sh $(ARGS)

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
	@echo "$(BLUE)▶$(NC) Building project..."
	@cmake --build $(BUILD_DIR) -j $(JOBS)
	@echo "$(GREEN)✓$(NC) Build complete"

clean: ## Clean build artifacts
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓$(NC) Clean complete"

compile-commands: $(BUILD_DIR)/CMakeCache.txt ## Link compile_commands.json for LSP support
	@ln -sf $(BUILD_DIR)/compile_commands.json .

# --- Distribution & MPI ---

generate-hostfile: ## Generate MPI hostfile from NODES variable
	@echo "$(GREEN)Generating hostfile...$(NC)"
	@rm -f $(HOSTFILE)
	@for node in $(NODES); do echo "$$node slots=1" >> $(HOSTFILE); done
	@cat $(HOSTFILE)

distribute: build ## Deploy binaries to nodes using rsync (Fast)
	@echo "$(BLUE)▶$(NC) Distributing binaries..."
	@for node in $(NODES); do \
		ssh $(REMOTE_USER)@$$node "mkdir -p $(REMOTE_DIR)" 2>/dev/null; \
		rsync -azP --delete $(BUILD_DIR)/bin $(BUILD_DIR)/lib $(REMOTE_USER)@$$node:$(REMOTE_DIR)/ 2>&1 | grep -v "^sending\|^sent\|^total" || true; \
	done
	@echo "$(GREEN)✓$(NC) Distribution complete"

check-nodes: ## Verify SSH connectivity to nodes
	@echo "$(BLUE)Checking node connectivity...$(NC)"
	@for node in $(NODES); do \
		printf "  %-15s " "$$node:"; \
		ssh -o ConnectTimeout=3 $(REMOTE_USER)@$$node "echo 'OK'" >/dev/null 2>&1 \
		&& echo "$(GREEN)✓$(NC)" || echo "✗"; \
	done

# --- Execution Wrappers ---

test: distribute ## Run distributed tests
	@./scripts/run_tests.sh $(ARGS)

benchmark: distribute ## Run benchmarks
	@./scripts/run_benchmark.sh $(ARGS)

profile: distribute ## Profile implementations
	@./scripts/run_profiling.sh $(ARGS)

run-distributed:
	@mpirun -np 3 --host localhost,10.0.0.2,10.0.1.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5 $(ARGS)

microbenchmark: distribute ## Run microbenchmark tool (use: make microbenchmark -- -n 1000 -i cuda)
	$(MAKE) run-distributed ARGS="$(REMOTE_DIR)/bin/microbenchmark $(ARGS)"


# --- Big Data Benchmark ---

# Defaults
NEURONS ?= 1000000
PINTRA ?= 10
PINTER ?= 1
STEPS ?= 5
OUTDIR ?= $(REMOTE_DIR)/bigdata/output
SEED ?= 123
MEM_LIMIT_GB ?= 0

# NODES and HOSTS are tricky. 
# If HOSTS is defined (comma sep), we use it for mpirun.
# We also need to derive a space-separated list for 'distribute'.
ifdef HOSTS
    MPI_HOSTS := $(HOSTS)
    DIST_NODES := $(shell echo $(HOSTS) | tr ',' ' ')
    NUM_NODES := $(shell echo $(HOSTS) | tr ',' '\n' | wc -l)
else
    # Fallback to existing NODES variable
    MPI_HOSTS := $(shell echo $(NODES) | tr ' ' ',')
    DIST_NODES := $(NODES)
    NUM_NODES := $(shell echo $(NODES) | wc -w)
endif

# Allow overriding count explicitly if needed (e.g. multiple ranks per node)
ifdef RANKS
    NUM_RANKS := $(RANKS)
else
    NUM_RANKS := $(NUM_NODES)
endif

bigdata-generate: ## Generate Big Data dataset
	@$(MAKE) distribute NODES="$(DIST_NODES)"
	@echo "$(BLUE)Generating Big Data dataset on $(NUM_RANKS) ranks...$(NC)"
	@./scripts/generate_bigdata.sh "$(DIST_NODES)" "$(OUTDIR)" "$(REMOTE_DIR)" "$(BUILD_DIR)" "$(NEURONS)" "$(MEM_LIMIT_GB)"

IMPL ?= optimized
PARTITIONER ?= linear

bigdata-run: ## Run Big Data simulation
	@$(MAKE) distribute NODES="$(DIST_NODES)"
	@echo "$(BLUE)Running Big Data simulation on $(NUM_RANKS) ranks ($(MPI_HOSTS))...$(NC)"
	@./scripts/run_bigdata.sh "$(DIST_NODES)" "$(OUTDIR)" "$(STEPS)" "$(REMOTE_DIR)" "$(IMPL)" "$(PARTITIONER)"
	@echo "$(GREEN)✓$(NC) Run complete. Results in $(OUTDIR)/results.json"

bigdata-verify: ## Verify distributed simulation against single-node run
	@echo "$(BLUE)Verifying Big Data simulation...$(NC)"
	@./scripts/verify_bigdata.sh "$(DIST_NODES)" "$(OUTDIR)" "$(STEPS)" "$(REMOTE_DIR)" "$(BUILD_DIR)" "$(NEURONS)"

START_NEURONS ?= 1000000
MULTIPLIER ?= 2

bigdata-capacity: ## Run capacity stress test
	@echo "$(BLUE)Running Capacity Stress Test...$(NC)"
	@./scripts/find_max_capacity.sh "$(START_NEURONS)" "$(MULTIPLIER)" "$(MPI_HOSTS)" "$(MEM_LIMIT_GB)"

bigdata-benchmark: ## Run comparison benchmark (Naive vs Optimized)
	@echo "$(BLUE)Running Big Data Benchmark...$(NC)"
	@python3 scripts/benchmark_bigdata.py
