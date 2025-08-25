# AceCoderV2

**Adversarial Code Generation System** - A multi-round pipeline for generating and evaluating adversarial test cases and programs.

## Quick Start with Docker (Recommended)

The easiest way to run AceCoderV2 is using Docker. This ensures consistent environment and dependencies.

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

### 1. Clone and Setup
```bash
git clone https://huggingface.co/datasets/siyiwu0330/acecoderv2
cd acecoderv2

# Copy environment template and configure
cp env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Run with Docker
```bash
# Build and start the application
./docker-start.sh build
./docker-start.sh start

# Or use docker-compose directly
docker-compose up -d
```

### 3. Access the Interface
- **Gradio Interface**: http://localhost:7860
- **View Logs**: `./docker-start.sh logs`

### Docker Commands
```bash
./docker-start.sh build    # Build the Docker image
./docker-start.sh start    # Start the application
./docker-start.sh stop     # Stop the application
./docker-start.sh logs     # View logs
./docker-start.sh dev      # Start in development mode
./docker-start.sh clean    # Clean up resources
```

## Development Setup

### Option 1: Conda Environment (Recommended)
```bash
# Create conda environment
conda env create -f environment-dev.yml
conda activate acecoderv2-dev

# Install project in development mode
pip install -e .

# Set up environment variables
cp env.example .env
# Edit .env and add your OpenAI API key

# Run the interface
python app.py
```

### Option 2: VS Code Dev Container
1. Install VS Code with Dev Containers extension
2. Open project in VS Code
3. Click "Reopen in Container" when prompted
4. Everything will be set up automatically!

### Option 3: UV/Local Development
```bash
# Install dependencies with uv
uv sync
uv pip install -e .

# Or use pip with requirements.txt
pip install -r requirements.txt
pip install git+https://github.com/TIGER-AI-Lab/AceCoder.git@dev
pip install -e .

# Run the interface
python app.py
```

For detailed conda setup instructions, see [CONDA_SETUP.md](CONDA_SETUP.md).

## Features

- **Multi-round Adversarial Generation**: Generate programs and test cases iteratively
- **Multiple Model Support**: GPT-4, GPT-3.5, and other OpenAI models
- **Real-time Visualization**: Interactive matrices showing program-test relationships
- **Configurable Parameters**: Control samples, tokens, rounds, and more
- **Pipeline Monitoring**: Real-time logs and progress tracking
- **Containerized**: Easy deployment with Docker

## Usage

### Web Interface
1. Start the application (Docker or local)
2. Open http://localhost:7860
3. Configure parameters:
   - **Rounds**: Number of adversarial rounds (1-50)
   - **Model**: OpenAI model to use
   - **Max Samples**: Number of questions to process (10-1000)
   - **Max Tokens**: Token limit per API call
4. Click "Start Pipeline" and monitor progress
5. View results in the Visualization tab

### Command Line
```bash
# Run pipeline directly
python advsersial_prompt/main.py \
  --rounds 3 \
  --model_name gpt-4.1-mini \
  --max_samples 50 \
  --output_dir outputs/my_experiment

# Run specific steps
python advsersial_prompt/step1_prompting.py --help
python advsersial_prompt/step2.1_openai_gen.py --help
```

## Project Structure

```
acecoderv2/
├── advsersial_prompt/          # Main pipeline code
│   ├── integrated_gradio_app.py # Web interface
│   ├── main.py                 # Pipeline orchestrator
│   ├── step1_prompting.py      # Problem generation
│   ├── step2.1_openai_gen.py   # Code generation
│   └── step2.2_eval.py         # Evaluation
├── code_eval/                  # Evaluation utilities
├── synthesizer/                # Data synthesis tools
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
└── outputs/                    # Generated results
```

## Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_MAX_TOKENS=4000
DEFAULT_MAX_SAMPLES=100
GRADIO_PORT=7860
```

### Pipeline Parameters
- **Rounds**: Number of adversarial iterations
- **Max Samples**: Questions to process per round
- **Max Tokens**: API call token limit
- **Model**: OpenAI model selection
- **Seed**: Random seed for reproducibility

## Evaluation

### Setup Evaluation Environment
```bash
mkdir -p eval
cd eval
git clone -b reasoning https://github.com/jdf-prog/LiveCodeBench
git clone https://github.com/jdf-prog/AceReasonEvalKit.git
```

### Legacy Synthesizer
```bash
cd synthesizer
bash scripts/run.sh

# For previous AceCoder dataset analysis
python ../scripts/format_old_acecoderv2_data
bash scripts/run_old_acecoderv2.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Use the Dev Container for consistent environment
4. Make your changes
5. Test with Docker
6. Submit a pull request

## License

See LICENSE file for details.

## Troubleshooting

### Common Issues
- **Port conflicts**: Change ports in `.env` file
- **API key errors**: Verify your OpenAI API key in `.env`
- **Memory issues**: Reduce `max_samples` parameter
- **Docker issues**: Run `./docker-start.sh clean` and rebuild

### Getting Help
- Check logs: `./docker-start.sh logs`
- View container status: `docker-compose ps`
- Reset environment: `./docker-start.sh clean && ./docker-start.sh build`
