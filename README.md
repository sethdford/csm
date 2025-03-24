# CSM (Conversational Speech Model)

A Rust implementation of a conversational speech model based on the LLaMA architecture. This project provides tools for processing audio input, generating speech responses, and managing model weights.

## Features

- Audio processing with WAV file support
- LLaMA-based transformer architecture
- Token-based audio generation
- Configurable model parameters
- CPU/GPU support via tch-rs

## Prerequisites

- Rust 1.70 or later
- CUDA toolkit (optional, for GPU support)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/csm.git
cd csm
```

2. Build the project:
```bash
cargo build --release
```

## Usage

1. Generate model weights:
```bash
cargo run --bin create_model
```

2. Generate test audio:
```bash
cargo run --example generate_test_audio
```

3. Process audio with the model:
```bash
cargo run --bin csm -- -w models/llama-1B.pth -f llama-1B -i audio/conversational_a.wav -o output.wav
```

## Command Line Arguments

- `-w, --model-path`: Path to model weights file
- `-f, --model-flavor`: Model flavor (e.g., llama-1B)
- `-i, --input-file`: Input audio file
- `-o, --output-file`: Output audio file
- `-m, --max-new-tokens`: Maximum number of new tokens to generate (default: 100)
- `-t, --temperature`: Temperature for sampling (default: 0.8)
- `-k, --top-k`: Top-k for sampling (default: 50)
- `-p, --top-p`: Top-p for sampling (default: 0.9)

## Project Structure

```
src/
├── audio/         # Audio processing module
├── models/        # Model implementations
├── utils/         # Utility functions
└── main.rs        # Main program entry point
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request