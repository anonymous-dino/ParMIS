#!/bin/bash

# ============================================================================
# ParMIS Execution Script
# ============================================================================
# This script compiles and runs ParMIS (v1 CPU, v1 GPU, and v2 GPU) on the
# available datasets (roadNet-CA and rgg_n_2_20_s0) with batches of 10^-6 and 10^-7.

set -e

# Configuration
DATASETS=("roadNet-CA" "rgg_n_2_20_s0")
BATCH_RATIOS=("6" "5")  # 6 = 10^-6, 5 = 10^-5
NUM_BATCHES=10
THREADS=64
GPU_ID=0

ROOT_DIR="/data/prajjwal/ParMIS"
BIN_DIR="${ROOT_DIR}/bin"
DATA_DIR="${ROOT_DIR}/datasets"

mkdir -p "$BIN_DIR"

# ============================================================================
# Compilation
# ============================================================================
echo "Compiling ParMIS executables..."

echo "Compiling ParMIS v1 CPU..."
g++ -fopenmp -O3 -o "$BIN_DIR/ParMIS_v1_cpu" "$ROOT_DIR/src/ParMIS_v1/cpu/ParMIS_v1_cpu.cpp"

echo "Compiling ParMIS v1 GPU..."
nvcc -O3 -arch=sm_70 -o "$BIN_DIR/ParMIS_v1_gpu" "$ROOT_DIR/src/ParMIS_v1/gpu/ParMIS_v1_gpu.cu" || 
nvcc -O3 -o "$BIN_DIR/ParMIS_v1_gpu" "$ROOT_DIR/src/ParMIS_v1/gpu/ParMIS_v1_gpu.cu"

echo "Compiling ParMIS v2 GPU..."
nvcc -O3 -arch=sm_70 -o "$BIN_DIR/ParMIS_v2_gpu" "$ROOT_DIR/src/ParMIS_v2/gpu/ParMIS_v2_gpu.cu" || 
nvcc -O3 -o "$BIN_DIR/ParMIS_v2_gpu" "$ROOT_DIR/src/ParMIS_v2/gpu/ParMIS_v2_gpu.cu"

echo "Compilation successful!"
echo "============================================================================"
echo

# ============================================================================
# Execution
# ============================================================================
for DATASET in "${DATASETS[@]}"; do
    MTX_FILE="${DATA_DIR}/converted_full/${DATASET}.mtx"
    MIS_FILE="${DATA_DIR}/MISs/${DATASET}.txt"

    if [ ! -f "$MTX_FILE" ] || [ ! -f "$MIS_FILE" ]; then
        echo "Skipping $DATASET: Missing input files (MTX or MIS)"
        continue
    fi

    for RATIO in "${BATCH_RATIOS[@]}"; do
        BATCH_DIR="${DATA_DIR}/Batches/${DATASET}/${RATIO}"
        
        if [ ! -d "$BATCH_DIR" ]; then
            echo "Skipping batch ratio $RATIO for $DATASET: Batch directory missing"
            continue
        fi

        echo "----------------------------------------------------------------------------"
        echo "Running tests for Graph: $DATASET | Batch Ratio Category: $RATIO (10^-$RATIO)"
        echo "----------------------------------------------------------------------------"

        # 1. ParMIS v1 CPU
        echo ">>> Running ParMIS v1 CPU..."
        echo "Command: $BIN_DIR/ParMIS_v1_cpu $MTX_FILE $MIS_FILE $BATCH_DIR $NUM_BATCHES $THREADS > ${ROOT_DIR}/results/ParMIS_v1_cpu/${DATASET}_${RATIO}.txt"
        time "$BIN_DIR/ParMIS_v1_cpu" "$MTX_FILE" "$MIS_FILE" "$BATCH_DIR" "$NUM_BATCHES" "$THREADS" > "${ROOT_DIR}/results/ParMIS_v1_cpu/${DATASET}_${RATIO}.txt" 2>&1
        echo

        # 2. ParMIS v1 GPU
        echo ">>> Running ParMIS v1 GPU..."
        echo "Command: $BIN_DIR/ParMIS_v1_gpu $MTX_FILE $MIS_FILE $BATCH_DIR $NUM_BATCHES $GPU_ID > ${ROOT_DIR}/results/ParMIS_v1_gpu/${DATASET}_${RATIO}.txt"
        time "$BIN_DIR/ParMIS_v1_gpu" "$MTX_FILE" "$MIS_FILE" "$BATCH_DIR" "$NUM_BATCHES" "$GPU_ID" > "${ROOT_DIR}/results/ParMIS_v1_gpu/${DATASET}_${RATIO}.txt" 2>&1
        echo

        # 3. ParMIS v2 GPU
        echo ">>> Running ParMIS v2 GPU..."
        echo "Command: $BIN_DIR/ParMIS_v2_gpu $MTX_FILE $MIS_FILE $BATCH_DIR $NUM_BATCHES > ${ROOT_DIR}/results/ParMIS_v2_gpu/${DATASET}_${RATIO}.txt"
        time "$BIN_DIR/ParMIS_v2_gpu" "$MTX_FILE" "$MIS_FILE" "$BATCH_DIR" "$NUM_BATCHES" > "${ROOT_DIR}/results/ParMIS_v2_gpu/${DATASET}_${RATIO}.txt" 2>&1
        echo
    done
done

echo "============================================================================"
echo "All executions completed successfully!"
