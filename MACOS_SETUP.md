# macOS Setup Guide for SAWN Project

This guide covers setup and running the SAWN littering detection system on macOS (Intel and Apple Silicon).

## Prerequisites

1. **macOS 11.0+** (Big Sur or later recommended for Apple Silicon)
2. **Python 3.9-3.11** (3.12 has some dependency issues)
3. **Homebrew** (package manager)
4. **Git**

## Installation Steps

### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.11

```bash
brew install python@3.11
```

### 3. Clone the Repository

```bash
cd ~/Projects  # or your preferred directory
git clone https://github.com/yourusername/sawn-project.git
cd sawn-project
```

### 4. Create Virtual Environment

```bash
# Using Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 5. Install Dependencies

#### For Apple Silicon (M1/M2/M3):

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

#### For Intel Macs:

```bash
# Standard installation
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**Note:** TensorFlow may not work well on Apple Silicon. Consider removing it if you're only using PyTorch models.

### 6. Download Dataset

```bash
# Create dataset directory
mkdir -p New_SawnDataset
cd New_SawnDataset

# Download your dataset (replace with your actual download method)
# curl -L "your-dataset-url" -o dataset.zip
# unzip dataset.zip
```

## Running the System

### Activate Virtual Environment (Always do this first)

```bash
source venv/bin/activate
```

### Training

```bash
python scripts/train_movinet.py \
    --data_dir New_SawnDataset \
    --epochs 30 \
    --lr 1e-4
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model models/movinet/movinet_best.pt \
    --test_dir New_SawnDataset
```

### Web Interface

```bash
python web/app.py
# Open browser to http://localhost:5000
```

### Full Pipeline

```bash
# Make script executable
chmod +x run_full_pipeline.sh

# Run pipeline
./run_full_pipeline.sh
```

## Performance Considerations

### Apple Silicon (M1/M2/M3) - MPS GPU Acceleration

- **Pros**:
  - Native GPU acceleration via Metal Performance Shaders (MPS)
  - Good performance for inference
  - Energy efficient
- **Cons**:
  - Training may be slower than NVIDIA GPUs
  - Some PyTorch operations not yet optimized for MPS

### Intel Macs

- **CPU Only**: Training will be significantly slower
- **Consider**: Using Google Colab or cloud GPU for training

### Check Your Device

```bash
python utils/device_utils.py
```

This will show:

- NVIDIA CUDA: Available on eGPU setups only
- Apple MPS: Available on M1/M2/M3 Macs
- CPU: Fallback for Intel Macs

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Make sure virtual environment is activated

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails

**Solution**: TensorFlow has issues on Apple Silicon. If you only use PyTorch models:

```bash
# Remove tensorflow from requirements.txt
pip install torch torchvision opencv-python flask scikit-learn matplotlib
```

### Issue: "torch.cuda" errors

**Solution**: This is normal on Mac. The code will automatically use MPS or CPU.

### Issue: Slow training on CPU

**Solution**:

- Reduce batch size: `--batch 2` or `--batch 4`
- Use fewer epochs for testing: `--epochs 10`
- Consider cloud GPU (Google Colab, AWS, etc.)

### Issue: Permission denied for .sh script

**Solution**:

```bash
chmod +x run_full_pipeline.sh
```

## GPU Performance Comparison

| Hardware        | Device | Training Speed  | Inference Speed |
| --------------- | ------ | --------------- | --------------- |
| NVIDIA RTX 3060 | CUDA   | Fast (baseline) | Very Fast       |
| M1 Max/Pro      | MPS    | ~70-80% of CUDA | Fast            |
| M1/M2 Base      | MPS    | ~50-60% of CUDA | Good            |
| Intel Mac       | CPU    | Very Slow       | Slow            |

## Recommended Workflow

### For Training:

1. **Apple Silicon Mac**: Train directly on Mac (decent speed)
2. **Intel Mac**: Use Google Colab with GPU runtime

### For Development & Testing:

- Any Mac works fine for code development
- Inference is fast enough on Apple Silicon

### For Production:

- Deploy on Linux server with NVIDIA GPU for best performance
- Or use Mac with MPS for smaller-scale deployments

## Additional Tools for Mac

### Install ffmpeg (for video processing):

```bash
brew install ffmpeg
```

### Install git-lfs (for large model files):

```bash
brew install git-lfs
git lfs install
```

## Environment Variables

Create `.env` file if needed:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback for unsupported MPS ops
export OMP_NUM_THREADS=4              # Optimize CPU usage
```

## Next Steps

1. Check device compatibility: `python utils/device_utils.py`
2. Validate dataset: `python validate_dataset_split.py`
3. Train model: `python scripts/train_movinet.py --data_dir New_SawnDataset`
4. Evaluate: `python analyze_evaluation.py`
5. Run web interface: `python web/app.py`

## Support

For Mac-specific issues:

- Check PyTorch MPS status: https://pytorch.org/docs/stable/notes/mps.html
- PyTorch forums: https://discuss.pytorch.org/

For project issues:

- Check README.md
- Open GitHub issue
