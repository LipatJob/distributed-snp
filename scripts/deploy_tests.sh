#!/bin/bash
# deploy_tests.sh - Deploy test executables and dependencies to remote hosts

set -e

# Configuration
REMOTE_HOST="10.0.0.2"
REMOTE_USER="${REMOTE_USER:-$(whoami)}"
PROJECT_DIR="/home/shared/distributed-snp-new"
BUILD_DIR="$PROJECT_DIR/build"

echo "=============================================="
echo "Deploying tests to distributed hosts"
echo "=============================================="

# 1. Create remote directory structure
echo "Creating remote directories on $REMOTE_HOST..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${BUILD_DIR} ${BUILD_DIR}/lib ${BUILD_DIR}/_deps/googletest-build/lib"

# 2. Copy test executables
echo "Copying test executables..."
scp ${BUILD_DIR}/test_matrix_ops ${REMOTE_USER}@${REMOTE_HOST}:${BUILD_DIR}/
scp ${BUILD_DIR}/test_snp_simulator ${REMOTE_USER}@${REMOTE_HOST}:${BUILD_DIR}/

# 3. Copy required libraries
echo "Copying shared libraries..."
scp ${BUILD_DIR}/lib/*.so* ${REMOTE_USER}@${REMOTE_HOST}:${BUILD_DIR}/lib/ 2>/dev/null || true
scp ${BUILD_DIR}/lib/*.a ${REMOTE_USER}@${REMOTE_HOST}:${BUILD_DIR}/lib/ 2>/dev/null || true

# 4. Copy GoogleTest libraries
echo "Copying GoogleTest libraries..."
scp ${BUILD_DIR}/_deps/googletest-build/lib/*.a ${REMOTE_USER}@${REMOTE_HOST}:${BUILD_DIR}/_deps/googletest-build/lib/ 2>/dev/null || true

# 5. Set permissions
echo "Setting executable permissions..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "chmod +x ${BUILD_DIR}/test_matrix_ops ${BUILD_DIR}/test_snp_simulator"

echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "You can now run distributed tests with:"
echo "  make test-distributed"
echo "or manually with:"
echo "  mpirun -np 2 --host localhost,${REMOTE_HOST} ${BUILD_DIR}/test_matrix_ops"
echo "  mpirun -np 2 --host localhost,${REMOTE_HOST} ${BUILD_DIR}/test_snp_simulator"
