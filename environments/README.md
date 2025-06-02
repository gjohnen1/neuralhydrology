# Neuralhydrology Environment Setup Guide

This directory contains conda environment files for different system configurations. Choose the appropriate environment based on your hardware and CUDA setup.

## Environment Files

### For CUDA-enabled Systems (Recommended)

**environment_rtx40_series.yml** - **BEST CHOICE for RTX 4090 and RTX 40-series GPUs**
- Optimized for RTX 40-series GPUs (RTX 4090, 4080, 4070, etc.)
- Uses PyTorch with CUDA 12.1 support via pip
- Tested and working on RTX 4090 with NVIDIA driver 535+
- Python 3.10

**environment_cuda12.yml** - For general CUDA 12.x systems
- Works with NVIDIA drivers 535+ and CUDA 12.x
- Uses PyTorch with CUDA 12.1 support via pip
- Python 3.10

**environment_cuda11_8.yml** - For older CUDA 11.8 systems
- For systems with NVIDIA drivers < 535
- Works with RTX 30-series and older GPUs
- Uses conda-installed PyTorch with CUDA 11.8
- Python 3.10

### For CPU-only Systems

**environment_cpu.yml** - CPU-only installation
- No GPU acceleration
- Suitable for testing or systems without NVIDIA GPUs
- Python 3.10

## Installation Instructions

### For RTX 4090 (Your Current Setup)

```bash
# Create the environment
conda env create -f environments/environment_rtx40_series.yml

# Activate the environment
conda activate neuralhydrology_rtx40

# Install neuralhydrology in development mode
pip install -e .
```

### For Other Systems

1. **Check your NVIDIA driver version:**
   ```bash
   nvidia-smi
   ```

2. **Choose the appropriate environment:**
   - Driver 535+ with RTX 40-series: `environment_rtx40_series.yml`
   - Driver 535+ with other GPUs: `environment_cuda12.yml`
   - Driver < 535: `environment_cuda11_8.yml`
   - No GPU: `environment_cpu.yml`

3. **Create and activate the environment:**
   ```bash
   conda env create -f environments/[chosen_environment].yml
   conda activate neuralhydrology  # or neuralhydrology_rtx40 for RTX40 series
   pip install -e .
   ```

## Troubleshooting

### CUDA Library Issues

If you encounter errors like `libcublasLt.so.11` not found:
1. Use the pip-based PyTorch installation (environment_rtx40_series.yml or environment_cuda12.yml)
2. Ensure your NVIDIA driver is up to date
3. Try the CUDA 11.8 environment as a fallback

### Environment Recreation

To recreate an environment from scratch:
```bash
conda env remove -n neuralhydrology  # or neuralhydrology_rtx40
conda env create -f environments/[chosen_environment].yml
conda activate [environment_name]
pip install -e .
```

## Testing Your Installation

After installation, test that everything works:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output for GPU systems:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090  # or your GPU model
```
