#!/bin/bash
# Enhanced memory monitoring version
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_HOME=$CONDA_PREFIX

echo "Starting training with memory monitoring..."

# Check available memory before starting
echo "System Memory:"
free -h
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Start background memory monitoring
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 5 > gpu_memory.log &
NVIDIA_PID=$!

free -m -s 5 > system_memory.log &
FREE_PID=$!

# Run training with timeout protection
timeout 3600 deepspeed --num_gpus=1 train.py \
    --deepspeed \
    --config examples/custom/wan22_squish_i2v_high_noise.toml \
    2>&1 | tee training_detailed.log

TRAIN_EXIT_CODE=$?

# Stop monitoring
kill $NVIDIA_PID 2>/dev/null
kill $FREE_PID 2>/dev/null

echo "Training exit code: $TRAIN_EXIT_CODE"
if [ $TRAIN_EXIT_CODE -eq 137 ] || [ $TRAIN_EXIT_CODE -eq 9 ]; then
    echo "Process killed by OOM killer - reduce memory usage"
fi

echo "Check gpu_memory.log and system_memory.log for memory usage patterns"