#!/bin/bash
# Ultra memory-optimized training for RTX 3090 following author's approach
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_HOME=$CONDA_PREFIX

# Aggressive memory conservation
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

echo "Starting ultra memory-optimized Wan2.2 training..."

# Check available memory
echo "System Memory:"
free -h
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Clear all caches
python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null

# Start monitoring
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 3 > gpu_memory_minimal.log &
NVIDIA_PID=$!

free -m -s 5 > system_memory_minimal.log &
FREE_PID=$!

echo "Training with author's memory optimization approach..."

# Use author's recommended approach
deepspeed --num_gpus=1 train.py \
    --deepspeed \
    --config _eugene/examples/wan22_squish_i2v_minimal_ram.toml \
    2>&1 | tee training_minimal_ram.log

TRAIN_EXIT_CODE=$?

# Stop monitoring
kill $NVIDIA_PID 2>/dev/null
kill $FREE_PID 2>/dev/null

echo "Training exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 137 ] || [ $TRAIN_EXIT_CODE -eq 9 ]; then
    echo "Still OOM - may need to:"
    echo "1. Further reduce video frames to 16-20"
    echo "2. Increase blocks_to_swap to 37-38"
    echo "3. Add more system RAM/swap"
elif [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Memory optimization successful!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi

echo "Memory usage logs: gpu_memory_minimal.log, system_memory_minimal.log"