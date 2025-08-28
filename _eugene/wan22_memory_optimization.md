# Wan2.2 I2V 14B Memory Issues on RTX 3090

## Problem

Training Wan2.2 I2V 14B (14.4B parameters) on RTX 3090 24GB + 32GB system RAM fails with:
- **System RAM exhaustion**: Model loading needs 24GB+ RAM
- **Process killed (exit code -9)**: OOM killer terminates training during model initialization  
- **GPU memory is fine**: Only uses ~5GB/24GB VRAM

## What We Did

**Memory optimizations made:**
- `blocks_to_swap = 36` (swap 36/40 model blocks to CPU)  
- `transformer_dtype = 'float8_e5m2'` (aggressive quantization)
- `rank = 16` (reduced LoRA parameters from 153M to 76M)
- `frame_buckets = [24]` (shorter videos, was 80 frames)
- `caching_batch_size = 1` (conservative dataset caching)

**Current status:**
- GPU memory: 5GB/24GB ✅ 
- Model loads successfully ✅
- DeepSpeed initializes ✅
- Still crashes during training setup with system RAM exhaustion ❌

## Why These Optimizations Work

**Based on our testing results:**

- **Block swapping (36/40)**: Successfully reduced GPU memory from 8GB+ to 5GB. Training logs show "Block swap enabled" message appears.
- **float8_e5m2 quantization**: Model loads without errors, previous attempts with int8 failed with "Invalid device string" error.
- **LoRA rank 16**: Reduced trainable params from 153M to 76M (50% cut). Training logs confirm the reduction.
- **24 frame videos**: Dataset caching completes successfully vs previous OOM during caching with 80 frames.
- **Conservative caching**: Prevents memory spikes during dataset processing phase.

**What still fails:**
- Memory monitoring shows system RAM drops from 24GB available to 374MB before crash
- Process gets killed with exit code -9 during training setup (after DeepSpeed initialization)
- GPU memory usage stays healthy at 5GB throughout

## Root Cause

The 14B model temporarily needs more than 32GB total memory during training initialization. System logs show:
- Available RAM: 24GB → 374MB  
- Swap usage: 2GB → completely full
- OOM killer activates when total memory (RAM + swap) exhausted

## Solution Needed

**Add more swap space** - the 2GB default isn't enough:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile  
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Files

- `_eugene/examples/wan22_squish_i2v_minimal_ram.toml`
- `_eugene/examples/squish_dataset_minimal.toml`  
- `_eugene/run_wan22_minimal_ram.sh`