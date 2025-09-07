# AceCoderV2 - Adversarial Code Generation System

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-siyiwu0330%2Facecoderv2-blue)](https://hub.docker.com/r/siyiwu0330/acecoderv2)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A comprehensive adversarial code generation system that creates challenging programming problems and evaluates AI models' coding capabilities through multi-round adversarial testing.

## 🚀 Quick Start

### Option 1: VS Code Dev Container (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/siyiwu0330/acecoderv2.git
   cd acecoderv2
   ```

2. **Open in VS Code Dev Container**:
   - Install the "Dev Containers" extension in VS Code
   - Open the project folder in VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select "Dev Containers: Reopen in Container"
   - Wait for the container to build and start

3. **Set your API key and run**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   python main.py --output_dir outputs/test --rounds 2 --max_samples 10
   ```

### Option 2: Docker

```bash
# Pull the image
docker pull siyiwu0330/acecoderv2:latest

# Run the container
docker run -d -p 7860:7860 --name acecoderv2 siyiwu0330/acecoderv2:latest

# Access the web interface
# Browser: http://localhost:7860
```

### Option 3: Local Installation

```bash
# Clone the repository
git clone https://github.com/siyiwu0330/acecoderv2.git
cd acecoderv2

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Run the pipeline
python main.py --output_dir outputs/test --rounds 1 --max_samples 10
```

## 🎯 Features

### Core Pipeline
- **Step 1: Prompting** - Transform simple problems into complex LeetCode-style problems
- **Step 1.1: Parsing** - Parse GPT responses and extract structured data
- **Step 2.1: Generation** - Generate multiple program solutions
- **Step 2.2: Evaluation** - Evaluate programs against test cases
- **Step 3: Filtering** - Filter and optimize test cases and programs
- **Step 4: Cross-Round Evaluation** - Comprehensive evaluation across all rounds

### Advanced Features
- **Skip Step 4** - Skip computationally expensive cross-round evaluation for large datasets
- **Parquet Conversion** - Convert results to efficient Parquet format
- **Hugging Face Integration** - Upload and download datasets from Hugging Face
- **Unified OpenAI Client** - Centralized API client management
- **Web Interface** - Interactive Gradio-based visualization

### Performance Optimizations
- **Parallel Processing** - Multi-threaded code evaluation
- **Memory Efficient** - Optimized for large datasets
- **Retry Logic** - Robust error handling with exponential backoff
- **Caching** - Intelligent caching of API responses

## 📊 Usage Examples

### Basic Pipeline
```bash
# Run 2 rounds with 10 samples
python main.py --output_dir outputs/test --rounds 2 --max_samples 10

# Skip step 4 for large datasets
python main.py --output_dir outputs/large --rounds 3 --max_samples 100 --skip_step4

# Use VLLM for generation
python main.py --output_dir outputs/vllm --use_vllm --rounds 1
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
# Start interactive interface
python integrated_gradio_app.py

# Start simple interface
python app.py
```

## 🏗️ Architecture

```
acecoderv2/
├── .devcontainer/          # VS Code dev container configuration
├── main.py                # Main pipeline (command-line)
├── main_full.py           # Full pipeline with all features
├── integrated_gradio_app.py  # Web interface
├── app.py                 # Simple web interface
├── openai_client.py       # Centralized OpenAI client
├── openai_utils.py        # Legacy OpenAI utilities
├── convert_to_parquet.py  # Parquet conversion tool
├── hf_dataset_converter.py # Hugging Face dataset converter
├── step1_prompting.py     # Problem transformation
├── step1.1_parsing.py     # Response parsing
├── step2.1_openai_gen.py  # Program generation
├── step2.1_vllm_gen.py    # VLLM generation
├── step2.2_eval.py        # Code evaluation
├── step_3_filter_tests.py # Test filtering
├── step4_cross_round_eval.py # Cross-round evaluation
└── ...                    # Other utilities
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
export GRADIO_SERVER_PORT="7860"  # Optional
export GRADIO_SERVER_NAME="0.0.0.0"  # Optional
```

### Command Line Options
```bash
python main.py --help
```

Key options:
- `--output_dir`: Output directory for results
- `--rounds`: Number of rounds to run
- `--max_samples`: Maximum number of samples
- `--skip_step4`: Skip cross-round evaluation
- `--use_vllm`: Use VLLM for generation
- `--verbose`: Enable verbose logging

## 📈 Performance

### Skip Step 4 Benefits
- **Solves hanging issues** with large problem sets (>50 problems)
- **Faster completion** by skipping computationally expensive evaluation
- **Memory efficient** with reduced resource usage
- **Stable execution** without timeouts

### Parquet Format Advantages
- **Storage efficiency** - more compact than JSON
- **Faster loading** - columnar storage for better performance
- **Memory friendly** - supports chunked reading
- **Tool compatibility** - works with pandas, Dask, etc.

## 🐳 Docker Support

### Available Images
- `siyiwu0330/acecoderv2:latest` - Latest version
- `siyiwu0330/acecoderv2:2.1.0` - Specific version

### Docker Compose
```yaml
version: '3.8'
services:
  acecoderv2:
    image: siyiwu0330/acecoderv2:latest
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=your-api-key-here
    volumes:
      - ./outputs:/home/acecoder/app/outputs
```

## 📚 Documentation

- [Skip Step 4 Usage](SKIP_STEP4_USAGE.md) - Guide for skipping cross-round evaluation
- [Parquet Conversion](PARQUET_CONVERSION_USAGE.md) - Data format conversion guide
- [Dev Container Setup](.devcontainer/README.md) - VS Code development environment
- [Changelog](CHANGELOG.md) - Version history and updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- Hugging Face for dataset hosting and tools
- The open-source community for various dependencies

## 📞 Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/siyiwu0330/acecoderv2/issues)
- **Docker Hub**: [Container images](https://hub.docker.com/r/siyiwu0330/acecoderv2)
- **Documentation**: Check individual script help with `--help`

## 🎉 Version 2.1.0 Highlights

- ✅ Skip Step 4 functionality for large datasets
- ✅ Parquet conversion tools
- ✅ Unified OpenAI client management
- ✅ VS Code dev container support
- ✅ Enhanced Docker integration
- ✅ Comprehensive documentation

---

**Made with ❤️ for the AI coding community**