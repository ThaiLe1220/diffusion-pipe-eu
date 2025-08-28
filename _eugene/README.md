# Eugene's Contributions to diffusion-pipe

This directory contains memory-optimized configurations and documentation for running large diffusion models on consumer hardware.

## Wan2.2 I2V 14B Memory Optimization

**Problem**: Training Wan2.2 I2V 14B fails on RTX 3090 24GB + 32GB RAM due to system memory exhaustion.

**Solution**: Aggressive memory optimization configurations that successfully initialize the model and reach training setup.

### Files:
- `examples/` - Memory-optimized TOML configurations
- `run_wan22_minimal_ram.sh` - Training script with memory monitoring  
- `wan22_memory_optimization.md` - Detailed documentation of the optimization process

### Usage:
```bash
cd /path/to/diffusion-pipe-eu
bash _eugene/run_wan22_minimal_ram.sh
```

### Status:
- ✅ Model loading successful
- ✅ GPU memory optimized (5GB/24GB)
- ✅ DeepSpeed initialization complete
- ❌ Requires additional swap space for training setup

See `wan22_memory_optimization.md` for complete details and troubleshooting.