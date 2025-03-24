# Contributing to CSM

Thank you for your interest in contributing to CSM! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/sethdford/csm.git
cd csm
```

2. Install dependencies:
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install FFmpeg (for audio processing)
# On macOS:
brew install ffmpeg
# On Ubuntu/Debian:
sudo apt-get install ffmpeg
```

3. Build the project:
```bash
cargo build
```

## Code Style

- Follow Rust's standard coding style
- Use `cargo fmt` to format your code
- Use `cargo clippy` to check for linting issues
- Keep functions focused and single-purpose
- Write clear, descriptive variable names
- Add comments for complex logic
- Include doc comments for public APIs

## Testing

- Write unit tests for new functionality
- Include integration tests for complex features
- Run the test suite before submitting:
```bash
cargo test
```

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "feat: description of your changes"
```

3. Push your changes:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request on GitHub

## Commit Messages

Follow conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Example:
```
feat(audio): add streaming audio processing support

- Implement audio streaming interface
- Add buffer management
- Update documentation
```

## Project Structure

- `src/audio/`: Audio processing and tokenization
- `src/models/`: Model implementations and architectures
- `src/utils/`: Utility functions and helpers
- `examples/`: Example code and usage
- `tests/`: Integration tests

## Performance Guidelines

- Profile code changes for performance impact
- Use appropriate data structures
- Minimize memory allocations
- Consider async operations for I/O
- Optimize tensor operations
- Use appropriate batch sizes

## Documentation

- Update README.md for significant changes
- Add inline documentation for complex code
- Include examples for new features
- Update API documentation
- Add comments for non-obvious optimizations

## Review Process

1. Ensure all tests pass
2. Address code review comments
3. Update documentation as needed
4. Squash commits if requested
5. Merge after approval

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in GitHub Discussions
- Check existing documentation
- Review closed issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 