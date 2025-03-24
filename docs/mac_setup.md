# Mac Setup Guide for CSM

This guide will help you set up the CSM (Conversational Speech Model) project on your Mac, specifically for Apple Silicon (M1/M2/M3) machines.

## Prerequisites

1. Xcode Command Line Tools
```bash
xcode-select --install
```

2. Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Environment Setup

1. Install Miniconda for Apple Silicon:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```
Follow the prompts to complete the installation. Make sure to initialize conda in your shell when asked.

2. Create and activate a new conda environment:
```bash
conda create -n csm python=3.10 -y
conda activate csm
```

3. Install PyTorch and related packages:
```bash
# Install PyTorch and related packages
conda install pytorch torchvision torchaudio -c pytorch -y

# Install additional dependencies
pip install tqdm requests regex fsspec packaging safetensors tokenizers
pip install transformers huggingface-hub --no-deps
```

## Project Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/csm.git
cd csm
```

2. Initialize the model:
```bash
# Create the models directory if it doesn't exist
mkdir -p models

# Initialize a random model for testing
cargo run --bin create_model
```

3. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

4. Build the project:
```bash
cargo build --release
```

## Running the Model

1. Generate test audio:
```bash
cargo run --example generate_test_audio
```

2. Process audio with the model:
```bash
cargo run --bin csm -- -w models/llama-1B.pth -f llama-1B -i conversational_a.wav -o output.wav
```

## Troubleshooting

### Common Issues

1. PyTorch Installation
If you encounter issues with PyTorch, try installing it with MPS (Metal Performance Shaders) support:
```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

2. Model Loading Errors
If you see errors about missing tensors or incorrect model format:
- Make sure you've run the `create_model` binary to initialize the model
- Check that the model weights file exists in the `models` directory
- Verify that you're using the correct model flavor flag (`-f llama-1B` or `-f llama-100M`)

3. Compilation Errors
Make sure you have the latest Xcode Command Line Tools installed and that your Rust toolchain is up to date:
```bash
rustup update
```

### Performance Optimization

For Apple Silicon Macs, the model can utilize the Metal Performance Shaders (MPS) backend for improved performance. To enable MPS:

1. Make sure you have PyTorch installed with MPS support
2. Use the `-c` flag to enable GPU acceleration when running the model

## Additional Resources

- [PyTorch Documentation for M1](https://pytorch.org/docs/stable/notes/mps.html)
- [Rust Installation Guide](https://www.rust-lang.org/tools/install)
- [HuggingFace Model Hub](https://huggingface.co/models)

## Support

If you encounter any issues not covered in this guide, please:
1. Check the project's issue tracker
2. Create a new issue with detailed information about your setup and the error you're encountering
3. Include the output of:
   - `cargo --version`
   - `python --version`
   - `conda list`
   - `rustc --version` 