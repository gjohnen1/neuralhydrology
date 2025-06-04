# Neuralhydrology Environment Setup Guide

This directory contains conda environment files for different system configurations. Choose the appropriate environment based on your hardware.

## Environment Files

### For GPU Systems (Recommended)

**environment_gpu.yml** - **GPU-accelerated training (RECOMMENDED)**
- For NVIDIA GPUs with CUDA support
- Compatible with modern GPUs (RTX 20/30/40 series, Tesla, etc.)
- Requires NVIDIA drivers version 450+ and CUDA 11.8+
- Uses PyTorch with CUDA 12.1 support
- Python 3.12

### For CPU-only Systems

**environment_cpu.yml** - **CPU-only training**
- No GPU acceleration required
- Works on any system (Windows, macOS, Linux)
- Suitable for testing or systems without NVIDIA GPUs
- Python 3.12

## Quick Start

### For GPU systems:
```bash
conda env create -f environments/environment_gpu.yml
conda activate neuralhydrology
```

### For CPU-only systems:
```bash
conda env create -f environments/environment_cpu.yml
conda activate neuralhydrology
```

## Installation Instructions

1. **Check your system:**
   - Run `nvidia-smi` to check if you have an NVIDIA GPU
   - If you see GPU information, use `environment_gpu.yml`
   - If not, use `environment_cpu.yml`

2. **Create the environment:**
   ```bash
   # For GPU systems
   conda env create -f environments/environment_gpu.yml
   
   # For CPU systems  
   conda env create -f environments/environment_cpu.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate neuralhydrology
   ```

4. **Install neuralhydrology package:**
   ```bash
   pip install -e .
   ```

5. **Verify installation:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

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
