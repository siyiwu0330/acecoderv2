# AceCoderV2 Development Container

This development container provides a complete environment for AceCoderV2 development with all dependencies pre-installed.

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd acecoderv2
   ```

2. **Open in VS Code Dev Container**:
   - Install the "Dev Containers" extension in VS Code
   - Open the project folder in VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select "Dev Containers: Reopen in Container"
   - Wait for the container to build and start

3. **Run the pipeline**:
   ```bash
   # Set your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"
   
   # Run the main pipeline
   python main.py --output_dir outputs/test --rounds 2 --max_samples 10
   
   # Or run with UI
   python integrated_gradio_app.py
   ```

## What's Included

### Pre-installed Software
- **Python 3.11** with all required packages
- **VS Code Extensions**:
  - Python support with Pylance
  - Jupyter notebook support
  - Code formatting (Black, isort)
  - Linting (pylint, flake8)
  - GitHub Copilot (if available)
  - JSON, YAML, Markdown support

### AceCoderV2 Features
- **Complete pipeline** with all steps
- **Skip Step 4** functionality for large datasets
- **Parquet conversion** tools
- **Unified OpenAI client** management
- **Gradio web interface** for visualization

### Development Tools
- **Git** and **GitHub CLI** for version control
- **Jupyter** for interactive development
- **Debugging** support with VS Code
- **Code formatting** and **linting** on save

## Available Commands

### Pipeline Execution
```bash
# Basic pipeline
python main.py --output_dir outputs/test --rounds 1 --max_samples 10

# Skip step 4 for large datasets
python main.py --output_dir outputs/large --rounds 3 --max_samples 100 --skip_step4

# Use VLLM for generation
python main.py --output_dir outputs/vllm --use_vllm --rounds 1

# Verbose output
python main.py --output_dir outputs/debug --verbose --rounds 1 --max_samples 5
```

### Data Processing
```bash
# Convert to Parquet format
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output

# Convert Hugging Face dataset
python hf_dataset_converter.py --dataset_name siyiwu0330/acecoderv2 --output converted.jsonl
```

### Web Interface
```bash
# Start Gradio interface
python integrated_gradio_app.py

# Start simple interface
python app.py
```

### Utilities
```bash
# Validate API key
python validate_api_key.py

# Test OpenAI client
python openai_client.py
```

## Port Forwarding

The following ports are automatically forwarded:
- **7860**: Gradio web interface
- **8000**: FastAPI (if used)
- **8888**: Jupyter notebooks
- **5000**: Flask (if used)

Access the web interface at: http://localhost:7860

## Environment Variables

Set these in your VS Code settings or in the terminal:

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
export GRADIO_SERVER_PORT="7860"  # Optional
export GRADIO_SERVER_NAME="0.0.0.0"  # Optional
```

## File Structure

```
acecoderv2/
├── .devcontainer/          # Dev container configuration
│   ├── devcontainer.json   # Main configuration
│   └── README.md          # This file
├── main.py                # Main pipeline (command-line)
├── main_full.py           # Full pipeline with all features
├── integrated_gradio_app.py  # Web interface
├── app.py                 # Simple web interface
├── openai_client.py       # Centralized OpenAI client
├── openai_utils.py        # Legacy OpenAI utilities
├── convert_to_parquet.py  # Parquet conversion tool
├── hf_dataset_converter.py # Hugging Face dataset converter
└── ...                    # Other pipeline files
```

## Development Workflow

1. **Start coding** in the container
2. **Run tests** with `python -m pytest`
3. **Format code** automatically on save (Black + isort)
4. **Lint code** with pylint and flake8
5. **Debug** using VS Code's debugger
6. **Commit changes** using Git integration

## Troubleshooting

### Container won't start
- Check Docker is running
- Try rebuilding: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### API key issues
- Set `OPENAI_API_KEY` environment variable
- Test with `python validate_api_key.py`

### Port conflicts
- Change ports in `devcontainer.json`
- Or stop conflicting services

### Missing dependencies
- All dependencies are pre-installed
- If something is missing, add it to `Dockerfile`

## Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check individual script help with `--help`
- **Docker Hub**: https://hub.docker.com/r/siyiwu0330/acecoderv2

## Version

This dev container is based on AceCoderV2 v2.1.0 with:
- Skip Step 4 functionality
- Parquet conversion tools
- Unified OpenAI client management
- Enhanced development experience
