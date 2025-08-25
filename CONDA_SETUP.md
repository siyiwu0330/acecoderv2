# Conda Environment Setup for AceCoderV2

This document provides instructions for setting up the AceCoderV2 environment using conda.

## Quick Setup (Recommended)

### 1. Create Environment from YAML

```bash
# Clone the repository
git clone https://huggingface.co/datasets/siyiwu0330/acecoderv2-new
cd acecoderv2-new

# Create and activate the environment
conda env create -f environment-dev.yml
conda activate acecoderv2-dev

# Install the project in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Test basic functionality
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"

# Test the main interface
python app.py --help
```

## Alternative Setup Methods

### Method 1: Manual Conda Environment

```bash
# Create base environment
conda create -n acecoderv2-dev python=3.10 -y
conda activate acecoderv2-dev

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia -y

# Install core scientific packages
conda install numpy pandas -c conda-forge -y

# Install Python dependencies
pip install -r requirements.txt

# Install AceCoder from source
pip install git+https://github.com/TIGER-AI-Lab/AceCoder.git@dev

# Install project in development mode
pip install -e .
```

### Method 2: Use Full Environment Export

If you have access to the original development environment:

```bash
# Use the complete environment file
conda env create -f environment.yml
conda activate acecoder
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp env.example .env

# Edit with your API keys
OPENAI_API_KEY=your_api_key_here
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_MAX_TOKENS=4000
DEFAULT_MAX_SAMPLES=100
GRADIO_PORT=7860
```

### GPU Support

For CUDA support, ensure you have:

1. **NVIDIA GPU** with compute capability 7.0+
2. **CUDA Toolkit 12.8** or compatible version
3. **Appropriate GPU drivers**

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Usage Examples

### 1. Start Gradio Interface

```bash
conda activate acecoderv2-dev
python app.py
```

Access at: http://localhost:7860

### 2. Run Pipeline Directly

```bash
# Run adversarial generation pipeline
python main.py \
  --rounds 3 \
  --model_name gpt-4.1-mini \
  --max_samples 50 \
  --output_dir outputs/my_experiment
```

### 3. Docker Alternative

If you prefer Docker over conda:

```bash
# Build and run with Docker
./docker-start.sh build
./docker-start.sh start
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvcc --version
   nvidia-smi
   
   # Reinstall PyTorch with correct CUDA version
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. **Memory Issues**
   - Reduce `max_samples` parameter
   - Use smaller models
   - Enable gradient checkpointing

3. **API Key Errors**
   - Verify `.env` file exists and contains valid API key
   - Check API key permissions and quotas

4. **Missing Dependencies**
   ```bash
   # Update all packages
   conda update --all
   
   # Reinstall specific package
   pip install --upgrade transformers
   ```

### Environment Updates

To update the environment:

```bash
# Export current environment
conda env export --no-builds > environment-updated.yml

# Update from file
conda env update -f environment-dev.yml --prune
```

## Development Workflow

### 1. Code Changes

```bash
# Activate environment
conda activate acecoderv2-dev

# Make your changes
# ...

# Test changes
python -m pytest tests/  # if tests exist
python app.py  # manual testing
```

### 2. Adding Dependencies

```bash
# Add conda dependencies
conda install new-package -c conda-forge

# Add pip dependencies
pip install new-package

# Update environment file
conda env export --no-builds > environment-dev.yml
```

### 3. Sharing Environment

```bash
# Create minimal environment file
conda env export --from-history > environment-minimal.yml

# Or create requirements.txt
pip freeze > requirements.txt
```

## Performance Tips

1. **Use conda-forge channel** for better package compatibility
2. **Pin major versions** to avoid conflicts
3. **Use mamba** for faster dependency resolution:
   ```bash
   conda install mamba -c conda-forge
   mamba env create -f environment-dev.yml
   ```

4. **Enable conda-libmamba-solver** for faster solving:
   ```bash
   conda install conda-libmamba-solver
   conda config --set solver libmamba
   ```

For more information, see the main [README.md](README.md) file.
