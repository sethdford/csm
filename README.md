# CSM (Conversational Speech Model)

A high-performance Rust implementation of a conversational speech model based on the LLaMA architecture. This project enables real-time audio processing and generation, making it ideal for applications requiring natural, context-aware speech interactions.

## Inspiration

This project is inspired by [Sesame](https://www.sesame.com/), who are pioneering the development of lifelike computer companions through natural voice interactions. Their work on crossing the uncanny valley of conversational voice has been particularly influential in shaping this implementation.

You can try their research demo at [Sesame's website](https://www.sesame.com/) to experience their vision of natural voice companions.

## Why Use CSM?

1. **Performance & Efficiency**
   - Written in Rust for maximum performance and memory safety
   - Leverages `tch-rs` for efficient tensor operations
   - Supports both CPU and GPU acceleration
   - Zero-cost abstractions and minimal runtime overhead

2. **Audio Processing Capabilities**
   - Real-time audio input/output processing
   - Multi-codebook audio tokenization for rich speech representation
   - High-quality WAV file support
   - Configurable audio parameters (sample rate, channels, bit depth)

3. **Advanced Model Architecture**
   - Based on the proven LLaMA transformer architecture
   - Multi-head attention with key-value caching
   - Rotary positional embeddings for better sequence modeling
   - Layer normalization and residual connections
   - Configurable model size and parameters

4. **Use Cases**
   - Voice assistants and chatbots
   - Real-time speech-to-speech translation
   - Audio content generation
   - Speech style transfer
   - Conversational AI applications
   - Audio data augmentation

## Features

- Audio processing with WAV file support
- LLaMA-based transformer architecture
- Token-based audio generation
- Configurable model parameters
- CPU/GPU support via tch-rs
- Efficient memory management
- Thread-safe design
- Comprehensive error handling

## Prerequisites

- Rust 1.70 or later
- CUDA toolkit (optional, for GPU support)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sethdford/csm.git
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
│   └── mod.rs     # WAV handling and tokenization
├── models/        # Model implementations
│   ├── mod.rs     # Model traits and common types
│   └── llama.rs   # LLaMA model implementation
├── utils/         # Utility functions
│   └── mod.rs     # Tensor operations and helpers
└── main.rs        # Main program entry point
```

## Performance Considerations

- Uses efficient tensor operations with `tch-rs`
- Implements streaming audio processing
- Optimized memory usage with zero-copy operations
- Supports batch processing for better throughput
- Configurable model size for different performance requirements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Sesame](https://www.sesame.com/) for their groundbreaking work in natural voice companions
- LLaMA model architecture by Meta AI
- PyTorch and tch-rs teams for the excellent tensor library
- Rust community for the amazing ecosystem
- Contributors and maintainers of all dependencies

## Related Projects

- [Sesame Research](https://www.sesame.com/research) - Original research on natural voice companions
- [Sesame Demo](https://www.sesame.com/demo) - Interactive demo of their voice technology