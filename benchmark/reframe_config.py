site_configuration = {
    'systems': [
        {
            'name': 'distributed-snp',
            'descr': 'Local distributed MPI+CUDA cluster',
            'hostnames': ['.+'],
            'modules_system': 'nomod', # Use 'lmod' if you use module load ...
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'mpirun',
                    'environs': ['cuda-gnu'],
                    # GLOBAL MPI FLAGS: These are injected into every mpirun command
                    'access': [
                        '--host localhost,10.0.0.2,10.0.1.2',
                        '--mca btl_tcp_if_include ens5',
                        '--mca oob_tcp_if_include ens5',
                        '--oversubscribe'
                    ],
                    'max_jobs': 1  # prevent oversubscribing if running multiple tests
                }
            ]
        }
    ],
    'environments': [
        {
            'name': 'cuda-gnu',
            'cc': 'mpicc',
            'cxx': 'mpic++',
            'ftn': 'mpif90',
        }
    ],
    'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_name)s: %(message)s',
                    'append': False
                }
            ]
        }
    ]
}