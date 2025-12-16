import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class SnpSimulationBenchmark(rfm.RunOnlyRegressionTest):
    sim_type = parameter(['cpu', 'sparse-cuda', 'optimized-cuda', 'naive-cuda-mpi', 'optimized-cuda-mpi'])
    input_size = parameter([256]) 
    partitioner = parameter(['linear', 'louvain', "redblue"])

    def __init__(self):
        # Environment setup
        self.valid_systems = ['distributed-snp:default']
        self.valid_prog_environs = ['cuda-gnu']
        
        self.sourcesdir = None

        # Path to the binary we compiled in Part 1
        self.executable = '/home/shared/tmp/distributed-snp-new/bin/sort_benchmark'
        
        # Set exclusive access to ensure stable benchmarking
        self.exclusive_access = True
    
    @run_after('init')
    def setup_configuration(self):
        # =================================================================
        # 1. Workflow Automation: Launch Configuration
        # =================================================================
        # Filter invalid combinations (e.g., partitioner doesn't matter for CPU)
        if self.sim_type != 'optimized-cuda-mpi' and self.partitioner != 'linear':
            self.skip_if(True, 'Partitioner only applies to Optimized MPI')
        if self.sim_type == 'cpu' and self.input_size > 1024:
            self.skip_if(True, 'CPU simulation only supports sizes up to 1024')

        # Configure MPI Ranks based on simulator type
        if 'mpi' in self.sim_type:
            self.num_tasks = 3         # Use 3 MPI ranks
            self.num_cpus_per_task = 1
        else:
            self.num_tasks = 1         # Serial / Single GPU
            self.num_cpus_per_task = 4 # Or however many threads you want

        # Build Command Line Arguments
        self.executable_opts = [
            f'--type {self.sim_type}',
            f'--size {self.input_size}',
            f'--part {self.partitioner}',
            f'--iter 10'
        ]

    @run_before('run')
    def set_launcher_options(self):
        self.job.launcher.options = [
            '--host localhost,10.0.0.2,10.0.1.2',
            '--mca btl_tcp_if_include ens5',
            '--mca oob_tcp_if_include ens5',
            '--oversubscribe'
        ]

    # =================================================================
    # 2. Metric Isolation & Validation
    # =================================================================
    @sanity_function
    def validate_output(self):
        # Ensure the run actually finished and passed verification
        return sn.assert_found(r'\[BenchMetric\] Verification: PASSED', self.stdout)

    @performance_function('ms')
    def loop_time(self):
        # Extract the precise LoopTime printed by Rank 0
        return sn.extractsingle(r'\[BenchMetric\] LoopTime: (\S+)', self.stdout, 1, float)

    @performance_function('steps/ms')
    def throughput(self):
        return sn.extractsingle(r'\[BenchMetric\] Throughput: (\S+)', self.stdout, 1, float)

    @performance_function('B')
    def mpi_bytes(self):
        # Optional: Only try to extract if it exists
        if 'mpi' not in self.sim_type:
            return 0.0
        return sn.extractsingle(r'\[BenchMetric\] MPI_Bytes: (\S+)', self.stdout, 1, float)

    @performance_function('ms')
    def compute_time(self):
        return sn.extractsingle(r'\[BenchMetric\] ComputeTime: (\S+)', self.stdout, 1, float)

    @performance_function('count')
    def steps(self):
        return sn.extractsingle(r'\[BenchMetric\] Steps: (\S+)', self.stdout, 1, float)

    @performance_function('count')
    def neurons(self):
        return sn.extractsingle(r'\[BenchMetric\] Neurons: (\S+)', self.stdout, 1, float)

    @performance_function('ms')
    def cuda_kernel_time(self):
        if 'cuda' not in self.sim_type:
            return 0.0
        return sn.extractsingle(r'\[BenchMetric\] CudaKernelTime: (\S+)', self.stdout, 1, float)

    @performance_function('ms')
    def mpi_comm_time(self):
        if 'mpi' not in self.sim_type:
            return 0.0
        return sn.extractsingle(r'\[BenchMetric\] MPI_CommTime: (\S+)', self.stdout, 1, float)