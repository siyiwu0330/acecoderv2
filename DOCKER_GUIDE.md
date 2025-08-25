# AceCoderV2 Complete Setup Guide (From Zero to Running)

## Complete Self-Contained Environment

AceCoderV2 provides a **completely self-contained Docker environment** with zero manual installation required after entering the container.

## 🚀 Step-by-Step Setup Guide

### Step 1: Environment Preparation

#### 1.1 Install Docker
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
# Re-login or run: newgrp docker

# macOS (using Homebrew)
brew install docker docker-compose
# Or download Docker Desktop: https://www.docker.com/products/docker-desktop

# Windows
# Download and install Docker Desktop: https://www.docker.com/products/docker-desktop
```

#### 1.2 Verify Docker Installation
```bash
docker --version
docker-compose --version
docker run hello-world
```

### Step 2: Get the Project

#### 2.1 Clone the Repository
```bash
# Method 1: Clone from Hugging Face
git clone https://huggingface.co/datasets/siyiwu0330/acecoderv2
cd acecoderv2

# IMPORTANT: The repository now uses main branch as default
# All complete code is available on the main branch
git checkout main  # This should be the default, but ensure you're on main

# Method 2: Download ZIP (if git not available)
# Visit https://huggingface.co/datasets/siyiwu0330/acecoderv2
# Click "Download repository" and extract
```

#### 2.2 Verify Project Structure
```bash
ls -la
# You should see:
# - Dockerfile (production environment)
# - Dockerfile.interactive (research environment)  
# - docker-compose.yml
# - docker-start.sh (management script)
# - env.example (environment template)
# - advsersial_prompt/ (main code)
# - README.md
```

### Step 3: Configure Environment

#### 3.1 Create Environment Configuration
```bash
# Copy environment template
cp env.example .env

# Edit configuration file
nano .env  # or use vim .env

# Add your OpenAI API key to .env file:
OPENAI_API_KEY=sk-your-api-key-here
```

#### 3.2 Verify Configuration
```bash
cat .env
# Ensure you see: OPENAI_API_KEY=your_actual_key
```

### Step 4: Choose and Start Environment

**Recommended: Research Environment** (most complete functionality)

#### 4.1 Build Research Environment Image
```bash
./docker-start.sh build_research
# This downloads base image and installs all dependencies (5-10 minutes)
```

#### 4.2 Start Research Environment
```bash
./docker-start.sh research
```

After successful startup, you'll see:
```
✅ Research environment started successfully!
Available interfaces:
- Gradio Interface: http://localhost:7861
- Jupyter Lab: http://localhost:8888

To access the interactive shell:
docker exec -it acecoderv2-research bash
```

### Step 5: Access and Use

#### 5.1 Option 1: Web Interface
Open browser and visit:
- **Gradio Interface**: http://localhost:7861
- **Jupyter Lab**: http://localhost:8888

#### 5.2 Option 2: Command Line Interface
```bash
# Enter container's interactive shell
./docker-start.sh shell

# You'll see welcome message and available commands
```

### Step 6: Start Using

#### 6.1 Using Gradio Web Interface
1. Open http://localhost:7861
2. In "Pipeline Control" tab:
   - Set **Rounds**: 1-3 (number of iterations)
   - Set **Max Samples**: 10-50 (number of questions, use small values for testing)
   - Select **Model**: gpt-4.1-mini
   - Confirm **OpenAI API Key** is filled
3. Click **"Start Pipeline"**
4. View results in **"Real-time Visualization"** tab

#### 6.2 Using Command Line
```bash
# Enter container
./docker-start.sh shell

# Check environment status
env-info

# Run complete pipeline
start-pipeline --rounds 2 --max_samples 20 --model_name gpt-4.1-mini

# Or use Python directly
python advsersial_prompt/main.py --rounds 2 --max_samples 20

# Start Gradio interface
start-gradio

# Start Jupyter Lab
start-notebook

# Run tests
run-tests
```

### Step 7: View Results

#### 7.1 Output File Locations
```bash
# Inside container, check:
ls -la outputs/
ls -la logs/

# Result files are typically in:
# outputs/acecoder_rounds/step2.2_eval_*.jsonl
# outputs/acecoder_rounds/visualizations/
```

#### 7.2 Visualization Results
- View in Gradio interface "Visualization" tab
- Or open notebooks/ directory in Jupyter Lab

### Step 8: Common Operations

#### 8.1 View Logs
```bash
# View application logs
./docker-start.sh logs

# View Docker logs
docker-compose logs -f
```

#### 8.2 Stop and Restart
```bash
# Stop all services
./docker-start.sh stop

# Restart
./docker-start.sh restart

# Clean up resources
./docker-start.sh clean
```

#### 8.3 Troubleshooting
```bash
# Check container status
docker ps -a

# Rebuild if issues occur
./docker-start.sh clean
./docker-start.sh build_research
./docker-start.sh research
```

## 🎯 Recommended First-Time Usage

### For Beginners (Simplest):
```bash
# 1. Install Docker
# 2. Clone project
git clone https://huggingface.co/datasets/siyiwu0330/acecoderv2
cd acecoderv2

# 3. Configure API key
echo "OPENAI_API_KEY=your_key_here" > .env

# 4. One-command startup
./docker-start.sh build_research
./docker-start.sh research

# 5. Open browser to http://localhost:7861
# 6. In web interface, use small parameters for first test:
#    - Rounds: 1
#    - Max Samples: 10
#    - Model: gpt-4.1-mini
```

### For Advanced Users:
```bash
# Use command line for more control
./docker-start.sh shell

# Inside container run:
start-pipeline --rounds 3 --max_samples 50 --model_name gpt-4.1-mini --output_dir outputs/my_experiment
```

## 📋 Complete Example Command Sequence

```bash
# === Complete Zero-to-Running Example ===

# 1. Clone project
git clone https://huggingface.co/datasets/siyiwu0330/acecoderv2
cd acecoderv2

# 2. Configure environment
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# 3. Build and start research environment
./docker-start.sh build_research
./docker-start.sh research

# 4. In another terminal, access container
./docker-start.sh shell

# 5. Inside container, check status
env-info

# 6. Run small-scale test
start-pipeline --rounds 1 --max_samples 5

# 7. View results
ls outputs/acecoder_rounds/

# 8. Start web interface (if you want GUI)
start-gradio
# Then visit http://localhost:7861
```

## 🔧 Available Environments

### Production Environment
- **Purpose**: Ready-to-use web interface
- **Includes**: Gradio interface auto-starts
- **Ports**: 7860 (Gradio), 8000 (alternative)
- **Usage**: `./docker-start.sh start`

### Research Environment
- **Purpose**: Interactive development and research
- **Includes**: 
  - Full Python data science stack (numpy, pandas, matplotlib, etc.)
  - Deep learning frameworks (PyTorch, Transformers)
  - Jupyter Lab environment
  - All evaluation tools pre-installed
  - Development tools (pytest, black, etc.)
- **Ports**: 7861 (Gradio), 8888 (Jupyter), 8001 (alternative)
- **Usage**: `./docker-start.sh research`

### Development Environment
- **Purpose**: Live code development
- **Includes**: Code mounting for real-time changes
- **Usage**: `./docker-start.sh dev`

## 📋 Pre-installed Components

### Core Dependencies
- ✅ Python 3.11 with comprehensive package ecosystem
- ✅ All project dependencies from uv.lock
- ✅ OpenAI, Anthropic, and other LLM API clients

### Data Science Stack
- ✅ NumPy, Pandas, Matplotlib, Seaborn, Plotly
- ✅ Scikit-learn, SciPy, Statsmodels
- ✅ PyTorch, Transformers, Tokenizers

### Development Tools
- ✅ Jupyter Lab, IPython, IPywidgets
- ✅ Pytest, Black, Isort, Flake8, MyPy
- ✅ Git, Vim, Nano, Tree, curl, wget

### Evaluation Environment
- ✅ LiveCodeBench (pre-cloned)
- ✅ AceReasonEvalKit (pre-cloned)
- ✅ All evaluation scripts and tools

### Web Frameworks
- ✅ Gradio, Streamlit, Flask, FastAPI
- ✅ Ready-to-use interfaces

## 🎯 Usage Scenarios

### For Researchers
```bash
# Start research environment
./docker-start.sh research

# Access interactive shell
./docker-start.sh shell

# Inside container - all commands ready:
start-gradio        # Launch web interface
start-notebook      # Launch Jupyter Lab
start-pipeline      # Run adversarial generation
run-tests          # Execute test suite
env-info           # Show environment status
```

### For Production Deployment
```bash
# Simple production deployment
./docker-start.sh build
./docker-start.sh start
# Web interface automatically available at http://localhost:7860
```

## 🌐 Remote Server Access

### Method 1: Gradio Share Link (Easiest!) 🌟

**AceCoderV2 automatically creates a public share link** - no configuration needed!

When you start the application, look for output like:
```
Running on public URL: https://xxxxx.gradio.live
```

This link works from anywhere in the world, with no firewall or network configuration required!

### Method 2: Direct Access (Advanced)
1. **Find your server's IP address**:
   ```bash
   curl ifconfig.me  # Shows your public IP
   ```

2. **Open firewall port** (if needed):
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 7860
   
   # CentOS/RHEL
   sudo firewall-cmd --add-port=7860/tcp --permanent
   sudo firewall-cmd --reload
   ```

3. **Access via browser**:
   - Visit: `http://YOUR_SERVER_IP:7860`
   - Replace `YOUR_SERVER_IP` with actual IP address

### Method 2: SSH Port Forwarding
If direct access isn't possible, use SSH tunneling:
```bash
# From your local machine
ssh -L 7860:localhost:7860 username@your_server_ip

# Then access http://localhost:7860 in your local browser
```

### Cloud Provider Notes:
- **AWS**: Configure Security Groups to allow inbound traffic on port 7860
- **Google Cloud**: Configure firewall rules for port 7860
- **Azure**: Configure Network Security Groups for port 7860
- **DigitalOcean**: Configure firewall settings in the control panel

### For Development
```bash
# Development with live code mounting
./docker-start.sh dev
# Code changes reflect immediately
```

## 🔍 Container Features

### Interactive Shell Commands
Once inside the container, you have convenient commands:

```bash
# Quick status check
env-info

# Start services
start-gradio                    # Web interface (port 7860)
start-notebook                  # Jupyter Lab (port 8888)
start-pipeline [args]           # Adversarial generation pipeline
run-tests [args]               # Test suite

# Direct Python usage
python advsersial_prompt/main.py --rounds 3 --max_samples 50
python advsersial_prompt/integrated_gradio_app.py
```

### Directory Structure
```
/workspace/              # Main working directory
├── advsersial_prompt/   # Core pipeline code
├── code_eval/           # Evaluation utilities
├── eval/                # Pre-installed evaluation tools
│   ├── LiveCodeBench/   # Benchmark suite
│   └── AceReasonEvalKit/ # Evaluation kit
├── outputs/             # Generated results (persistent)
├── logs/                # Application logs (persistent)
├── experiments/         # Your experiments (persistent)
└── notebooks/           # Jupyter notebooks (persistent)
```

### Persistent Data
All important data is persisted in Docker volumes:
- `outputs/` - Generated results
- `logs/` - Application logs  
- `experiments/` - Your experiment data
- `notebooks/` - Jupyter notebooks
- `models/` - Downloaded models

## 🛠️ Advanced Usage

### Accessing Running Containers
```bash
# Check running containers
docker ps

# Access shell in any running container
./docker-start.sh shell

# Or directly
docker exec -it acecoderv2-research bash
docker exec -it acecoderv2-app bash
```

### View Logs
```bash
# Application logs
./docker-start.sh logs

# Docker compose logs
docker-compose logs -f
```

### Multiple Environments
```bash
# Run both production and research simultaneously
./docker-start.sh start                    # Production on 7860
./docker-start.sh research                 # Research on 7861, 8888
```

### Data Management
```bash
# Backup your data
docker volume ls                           # List volumes
docker run --rm -v acecoderv2-outputs:/data ubuntu tar czf - /data > outputs_backup.tar.gz

# Restore data
docker run --rm -v acecoderv2-outputs:/data ubuntu tar xzf - < outputs_backup.tar.gz
```

## 🔧 Customization

### Environment Variables
Add to your `.env` file:
```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional customization
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_MAX_TOKENS=4000
DEFAULT_MAX_SAMPLES=100
GRADIO_PORT=7860
```

### Custom Configurations
The research environment includes configuration files:
- `~/.bashrc` - Shell aliases and environment
- Jupyter Lab configurations
- Git configurations

## ✅ Benefits of This Approach

1. **Zero Setup Time**: Everything pre-installed and configured
2. **Reproducible**: Same environment on any system
3. **Isolated**: No conflicts with host system
4. **Complete**: All tools and dependencies included
5. **Persistent**: Your work is saved across container restarts
6. **Scalable**: Easy to deploy on any Docker-capable system

## 🆘 Troubleshooting

### Common Issues
```bash
# Container won't start
./docker-start.sh clean
./docker-start.sh build

# Port conflicts
# Edit .env file to change ports

# Out of disk space
docker system prune -a

# Permission issues
# Check that Docker daemon is running
```

### Getting Help
```bash
# Check container status
docker ps -a

# View container logs
docker logs acecoderv2-app
docker logs acecoderv2-research

# Access environment info
./docker-start.sh shell
env-info
```

## ⏱️ Time Expectations

- **Docker Installation**: 5-10 minutes (one-time)
- **Project Clone**: 1-2 minutes
- **Environment Build**: 5-10 minutes (one-time)
- **First Run**: 2-3 minutes
- **Total Setup Time**: ~15-20 minutes

## 💡 Tips for Success

### First-Time Users
1. **Start Small**: Use `--max_samples 5` for your first test
2. **Monitor Costs**: Each API call costs money, start with small experiments
3. **Check Logs**: Use `./docker-start.sh logs` if something goes wrong
4. **Save Work**: All outputs are automatically saved in `outputs/` directory

### Advanced Usage
1. **Multiple Experiments**: Use different `--output_dir` for each experiment
2. **Batch Processing**: Run multiple rounds with different parameters
3. **Custom Models**: Experiment with different OpenAI models
4. **Visualization**: Use Jupyter Lab for custom analysis of results

### Performance Optimization
1. **Parallel Processing**: Increase `--num_proc` for faster evaluation
2. **Batch Size**: Adjust batch sizes for memory optimization
3. **Model Selection**: Choose appropriate models for your use case

## 🔍 What Happens During First Run

When you run the pipeline for the first time:

1. **Step 1**: Generates initial problems and test cases from dataset
2. **Step 2.1**: Creates adversarial programs using OpenAI API
3. **Step 2.2**: Evaluates programs against test cases
4. **Step 3**: Filters and processes results
5. **Visualization**: Generates interactive matrices and plots

Expected output files:
```
outputs/acecoder_rounds/
├── step1_prompting_results.jsonl
├── step2.1_gen_*.jsonl
├── step2.2_eval_*.jsonl
├── step_3_filter_tests_*.jsonl
└── visualizations/
    ├── visualization_history.jsonl
    └── cross_round_evaluation.jsonl
```

## 🚨 Common Gotchas

1. **API Key**: Make sure your OpenAI API key is valid and has credits
2. **Network**: Ensure stable internet connection for API calls
3. **Disk Space**: Large experiments can generate significant data
4. **Memory**: Reduce `max_samples` if you encounter memory issues
5. **Ports**: Make sure ports 7861 and 8888 are not in use

## 🎉 Success Indicators

You know everything is working when:
- ✅ Docker containers start without errors
- ✅ Web interfaces are accessible (7861, 8888)
- ✅ API calls to OpenAI succeed
- ✅ Results appear in `outputs/` directory
- ✅ Visualizations show in Gradio interface

This complete setup ensures that anyone can run the AceCoderV2 system with minimal effort while having access to a complete, professional research environment.
