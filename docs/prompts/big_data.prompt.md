# Big-Data Distributed SNP Benchmark Prompt

## Objective
Add an end-to-end workflow to benchmark the **distributed SNP simulator (MPI + CUDA)** across multiple nodes using a synthetic SNP system dataset that is **too large to fit into the memory of a single machine**.

The workflow must support:
1) generating a large synthetic dataset that can be partitioned across nodes,
2) running a short distributed simulation where **each node loads only its local partition from a local file**,
3) collecting structured performance metrics to analyze scalability.

## Hard requirements (must implement)

### A) Synthetic dataset generation
- Generate a synthetic SNP system whose total size is user-configurable (e.g., neuron count / matrix sizes / density).
- Partition the dataset across `N` nodes.
- Produce a **descriptor file** that defines:
	- global system size (and any other global parameters required to interpret files),
	- connectivity parameters,
	- the mapping from node/rank/host to its partition file path.
- Each node must be able to load its own partition solely from its **local** partition file.

### B) Connectivity structure (topology-aware)
- **Within-node partitions should be strongly connected** (higher edge density / connectivity inside each partition).
- **Between-node connections should be sparse** (low cross-partition connectivity).
- The descriptor (or a companion report) must include summary stats that make this verifiable (e.g., intra-edge count vs inter-edge count).

### C) “Too big for one machine” check
- Provide a testing script (or executable) that verifies the dataset cannot fit into the memory of a single machine.
- It is acceptable to implement this as a **size-based check**:
	- compute total dataset size from actual file sizes + metadata,
	- compare against detected system RAM or a user-provided memory limit (e.g., `MEM_LIMIT_GB`).
- The check must be runnable as part of the workflow and must fail if the dataset is too small.

Additionally, because the simulator is CUDA-accelerated:
- Provide an optional (but recommended) GPU-side capacity check that estimates the per-rank working-set size and compares it to a user-provided `GPU_MEM_LIMIT_GB` (or detected GPU memory when available).
- This check should ensure the dataset is “big” in the intended way (global dataset is too large for one machine), without requiring each rank to exceed a single GPU’s memory.

### D) Distributed run via MPI + CUDA (short run)
- Run the distributed SNP simulator across specified nodes using MPI and execute compute kernels on GPUs via CUDA.
- Each rank/node must:
	- read the descriptor,
	- locate its assigned partition,
	- load only that partition,
	- select/bind to a GPU device (one GPU per rank, unless the project already supports a different mapping),
	- participate in the distributed simulation.
- The simulation should run only a small number of steps (e.g., 3–10) to keep runtime manageable.

GPU execution requirements:
- Make the GPU selection deterministic per rank (e.g., `rank % local_gpu_count`) and/or support `CUDA_VISIBLE_DEVICES`.
- If multiple ranks run on the same node, ensure they do not all attempt to use the same GPU unless explicitly configured.

### E) Configurability
- The user must be able to specify:
	- number of nodes (`NODES`)
	- hostnames/IP addresses (either `HOSTS="h1,h2,..."` or `HOSTFILE=...`)
	- dataset size for generation (e.g., `NEURONS=...`, and connectivity params)

### F) Scaling experiment (2 nodes → 3 nodes)
- Provide an experiment that increases nodes from **2 to 3** while keeping the **total dataset size constant**.
- The output must allow comparing scalability when adding resources.

### G) Performance metrics and structured logs
- During the distributed run, collect and log at least:
	- total execution time
	- MPI communication overhead (time spent communicating)
	- CUDA compute/kernel time (if available)
	- nodes, dataset size, steps
- Write logs in **JSON or CSV** (choose one) with a stable schema.
- Logs must be saved to a predictable location under `benchmark/` or `output/`.

If available in the current codebase, include both:
- `mpi_comm_time_ms`
- `cuda_kernel_time_ms`

## UX requirement: exactly two Make targets
The user must be able to run this benchmark with exactly two Makefile targets:

1) `make bigdata-generate`
	 - compiles/sets up anything needed
	 - generates the synthetic dataset partitions
	 - writes the descriptor file
	 - runs the “too big for one machine” check (or provides a clearly documented target/step that is executed automatically)

2) `make bigdata-run`
	 - compiles/sets up anything needed
	 - runs the MPI distributed simulation across the specified hosts (MPI + CUDA)
	 - writes the structured performance metrics log(s)
	 - runs the 2-node and 3-node experiment (or provides a single invocation that can be repeated with `NODES=2` and `NODES=3`)

The Makefile must handle all necessary compilation and MPI execution details.

## Inputs (recommended interface)
Support configuring via Make variables and/or environment variables:
- `OUTDIR=output/bigdata`
- `NODES=2`
- `HOSTS="host1,host2"`
- `NEURONS=...`
- `PINTRA=...` (within-node connectivity)
- `PINTER=...` (between-node connectivity)
- `STEPS=5`
- `MEM_LIMIT_GB=...`
- `GPU_MEM_LIMIT_GB=...`
- `SEED=123`

## Acceptance criteria (definition of done)
- Source code in the ./bigdata/ directory implementing the above functionality.
- `make bigdata-generate` produces a descriptor and per-node partition files.
- The generated dataset is large enough to fail the single-machine memory check (unless the user lowers the threshold).
- `make bigdata-run` runs a short MPI distributed simulation where each rank loads only its local partition.
- Running with `NODES=2` and then `NODES=3` with constant total dataset size produces comparable structured logs.
- The logs include total time and MPI communication overhead, and can be used to assess scaling.
- Results must be created in the ./bigdata/results/ directory.

## Reference files
- `src/snp/ISnpSimulator.hpp` (for simulator interface)
- `src/snp/OptimizedCudaMpiSnpSimulator.cu` (for existing MPI + CUDA simulator)
- `Makefile` (for build and MPI execution patterns)
- `tests/snp/SnpSimulatorTest.cpp` (for existing test patterns)
- `src/snp/SnpSystemConfig.hpp` (for SNP system representation. May need extension to efficiently support partitioned loading and big data)

## Notes / constraints
- Keep implementation minimal and aligned with existing project architecture (interfaces, factories, build system).
- Prefer deterministic generation via a seed.
- Avoid adding extra modes, UIs, or unrelated refactors.