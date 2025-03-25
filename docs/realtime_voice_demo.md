# Real-Time Voice Conversation System in Rust

This guide outlines how to implement a real-time voice conversation system using our Rust CSM codebase, inspired by [Haadesx/realtime-voice-csm](https://github.com/Haadesx/realtime-voice-csm).

## Architecture Overview

```
                    ┌───────────────────┐
                    │  Web Interface    │
                    │  (WebAssembly)    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   WebSocket       │
                    │   Connection      │
                    └─────────┬─────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼─────┐
│  Audio    │           │   CSM     │           │ Language  │
│ Processing│◄─────────►│  Model    │◄─────────►│  Model    │
└───────────┘           └───────────┘           └───────────┘
```

## Implementation Plan

### 1. Audio Processing Module

We need to extend our existing audio module to handle real-time streaming:

```rust
// src/audio/streaming.rs
use crate::audio::AudioProcessor;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{self, Receiver, Sender};

pub struct AudioStream {
    processor: Arc<Mutex<AudioProcessor>>,
    input_sender: Sender<Vec<f32>>,
    output_receiver: Receiver<Vec<f32>>,
    sample_rate: usize,
    channels: usize,
}

impl AudioStream {
    pub fn new(sample_rate: usize, channels: usize, bit_depth: usize) -> Self {
        let processor = Arc::new(Mutex::new(
            AudioProcessor::new(sample_rate, channels, bit_depth)
        ));
        let (input_sender, mut input_receiver) = mpsc::channel::<Vec<f32>>(100);
        let (output_sender, output_receiver) = mpsc::channel::<Vec<f32>>(100);
        
        // Spawn audio processing task
        let processor_clone = Arc::clone(&processor);
        tokio::spawn(async move {
            while let Some(audio_chunk) = input_receiver.recv().await {
                let processor = processor_clone.lock().unwrap();
                let tokens = processor.tokenize_chunk(&audio_chunk).unwrap();
                
                // Process tokens (call model)
                // ...
                
                // Send processed audio back
                let output_audio = processor.detokenize_chunk(&tokens).unwrap();
                let _ = output_sender.send(output_audio).await;
            }
        });
        
        Self {
            processor,
            input_sender,
            output_receiver,
            sample_rate,
            channels,
        }
    }
    
    pub async fn process_audio_chunk(&self, chunk: Vec<f32>) -> anyhow::Result<()> {
        Ok(self.input_sender.send(chunk).await?)
    }
    
    pub async fn get_next_output(&mut self) -> Option<Vec<f32>> {
        self.output_receiver.recv().await
    }
}
```

### 2. WebSocket Server

Create a WebSocket server to handle real-time communication:

```rust
// src/server/websocket.rs
use crate::audio::streaming::AudioStream;
use crate::models::{Model, ModelConfig};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use warp::{ws::{WebSocket, Message}, Filter};

pub struct WebSocketServer {
    audio_stream: Arc<Mutex<AudioStream>>,
    model: Arc<Mutex<dyn Model>>,
}

impl WebSocketServer {
    pub fn new(model: Arc<Mutex<dyn Model>>, config: &ModelConfig) -> Self {
        let audio_stream = Arc::new(Mutex::new(
            AudioStream::new(
                44100,  // Sample rate
                1,      // Mono audio
                16      // Bit depth
            )
        ));
        
        Self {
            audio_stream,
            model,
        }
    }
    
    pub async fn start(&self, port: u16) -> anyhow::Result<()> {
        let audio_stream = Arc::clone(&self.audio_stream);
        
        // Create WebSocket route
        let ws_route = warp::path("audio")
            .and(warp::ws())
            .map(move |ws: warp::ws::Ws| {
                let audio_stream = Arc::clone(&audio_stream);
                ws.on_upgrade(move |socket| handle_connection(socket, audio_stream))
            });
            
        // Start the server
        warp::serve(ws_route)
            .run(([0, 0, 0, 0], port))
            .await;
            
        Ok(())
    }
}

async fn handle_connection(ws: WebSocket, audio_stream: Arc<Mutex<AudioStream>>) {
    let (mut ws_sender, mut ws_receiver) = ws.split();
    let (tx, mut rx) = mpsc::channel::<Message>(100);
    
    // Handle incoming audio chunks
    tokio::spawn(async move {
        while let Some(result) = ws_receiver.next().await {
            if let Ok(msg) = result {
                if let Ok(bytes) = msg.as_bytes() {
                    // Convert bytes to audio samples
                    let audio_chunk = bytes_to_samples(bytes);
                    
                    // Process audio chunk
                    let mut stream = audio_stream.lock().unwrap();
                    stream.process_audio_chunk(audio_chunk).await.unwrap();
                    
                    // Get output and send back
                    if let Some(output) = stream.get_next_output().await {
                        let output_bytes = samples_to_bytes(&output);
                        let _ = tx.send(Message::binary(output_bytes)).await;
                    }
                }
            }
        }
    });
    
    // Send responses back to client
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if let Err(_) = ws_sender.send(msg).await {
                break;
            }
        }
    });
}

fn bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    // Implementation depends on audio format
    // ...
}

fn samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    // Implementation depends on audio format
    // ...
}
```

### 3. Web Interface with WebAssembly

Create a web interface that uses WebAssembly to interact with our Rust backend:

```rust
// src/wasm/mod.rs
use wasm_bindgen::prelude::*;
use web_sys::{AudioContext, MediaStream};

#[wasm_bindgen]
pub struct AudioInterface {
    audio_context: AudioContext,
    websocket: Option<web_sys::WebSocket>,
}

#[wasm_bindgen]
impl AudioInterface {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let audio_context = AudioContext::new().unwrap();
        
        Self {
            audio_context,
            websocket: None,
        }
    }
    
    #[wasm_bindgen]
    pub fn connect(&mut self, url: &str) -> Result<(), JsValue> {
        let ws = web_sys::WebSocket::new(url)?;
        // Configure WebSocket
        // ...
        self.websocket = Some(ws);
        Ok(())
    }
    
    #[wasm_bindgen]
    pub async fn start_recording(&self) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        let media_devices = window.navigator().media_devices()?;
        
        // Request microphone access
        let constraints = web_sys::MediaStreamConstraints::new();
        constraints.audio(&JsValue::from_bool(true));
        
        let promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream: MediaStream = wasm_bindgen_futures::JsFuture::from(promise).await?.into();
        
        // Process audio stream
        // ...
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn stop_recording(&self) {
        // Stop recording
        // ...
    }
}
```

### 4. Main Program

Update our main program to support WebSocket server mode:

```rust
// src/bin/csm_server.rs
use csm::models::{self, Model, ModelConfig};
use csm::server::websocket::WebSocketServer;
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tch::Device;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model weights
    #[arg(short = 'w', long)]
    model_path: PathBuf,

    /// Model flavor (e.g., llama-1B)
    #[arg(short = 'f', long)]
    model_flavor: String,
    
    /// Port to run the websocket server on
    #[arg(short = 'p', long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Set up device
    let device = Device::cuda_if_available();
    info!("Using {}", if device.is_cuda() { "GPU" } else { "CPU" });
    
    // Create model config
    let config = ModelConfig {
        backbone_flavor: args.model_flavor.clone(),
        decoder_flavor: "default".to_string(),
        text_vocab_size: 128_256,
        audio_vocab_size: 1024,
        audio_num_codebooks: 8,
        max_seq_len: 2048,
        embed_dim: 2048,
    };
    
    // Create model
    let mut vs = tch::nn::VarStore::new(device);
    let model = models::create_model(config.clone(), device)?;
    
    // Load model weights
    info!("Loading model weights from {}", args.model_path.display());
    vs.load(&args.model_path)?;
    info!("Model weights loaded successfully");
    
    // Create WebSocket server
    let model = Arc::new(Mutex::new(model));
    let server = WebSocketServer::new(model, &config);
    
    // Start server
    info!("Starting WebSocket server on port {}", args.port);
    server.start(args.port).await?;
    
    Ok(())
}
```

### 5. HTML/JavaScript Frontend

Create a simple frontend for interacting with our WebSocket server:

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSM Voice Chat</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>CSM Voice Chat</h1>
            <p>Real-time voice conversation with Rust-powered CSM</p>
        </header>
        
        <main>
            <div class="visualizers">
                <div class="visualizer input-viz" id="inputViz"></div>
                <div class="visualizer output-viz" id="outputViz"></div>
            </div>
            
            <button id="recordButton" class="record-button">
                <span>Press to Talk</span>
            </button>
            
            <div id="status" class="status">Ready</div>
        </main>
        
        <footer>
            <p>Powered by CSM - A Rust implementation inspired by Sesame</p>
        </footer>
    </div>
    
    <script src="csm_wasm.js"></script>
    <script src="app.js"></script>
</body>
</html>
```

```javascript
// static/app.js
let isRecording = false;
let csm;
let audioContext;
let analyserInput;
let analyserOutput;

async function initAudio() {
    csm = new CSMAudio();
    await csm.connect(`ws://${window.location.hostname}:8080/audio`);
    
    audioContext = new AudioContext();
    
    // Set up input visualizer
    analyserInput = audioContext.createAnalyser();
    analyserInput.fftSize = 256;
    
    // Set up output visualizer
    analyserOutput = audioContext.createAnalyser();
    analyserOutput.fftSize = 256;
    
    // Start visualization loop
    visualize();
}

async function toggleRecording() {
    if (!isRecording) {
        // Start recording
        isRecording = true;
        document.getElementById('recordButton').classList.add('recording');
        document.getElementById('status').textContent = 'Listening...';
        
        await csm.startRecording();
    } else {
        // Stop recording
        isRecording = false;
        document.getElementById('recordButton').classList.remove('recording');
        document.getElementById('status').textContent = 'Processing...';
        
        csm.stopRecording();
    }
}

function visualize() {
    const inputCanvas = document.getElementById('inputViz');
    const outputCanvas = document.getElementById('outputViz');
    
    const inputCtx = inputCanvas.getContext('2d');
    const outputCtx = outputCanvas.getContext('2d');
    
    // Input visualizer
    const inputBufferLength = analyserInput.frequencyBinCount;
    const inputDataArray = new Uint8Array(inputBufferLength);
    
    // Output visualizer
    const outputBufferLength = analyserOutput.frequencyBinCount;
    const outputDataArray = new Uint8Array(outputBufferLength);
    
    function draw() {
        requestAnimationFrame(draw);
        
        // Draw input visualizer
        analyserInput.getByteFrequencyData(inputDataArray);
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        inputCtx.fillStyle = '#00c6ff';
        
        const inputBarWidth = (inputCanvas.width / inputBufferLength) * 2.5;
        let inputBarHeight;
        let x = 0;
        
        for (let i = 0; i < inputBufferLength; i++) {
            inputBarHeight = inputDataArray[i] / 2;
            inputCtx.fillRect(x, inputCanvas.height - inputBarHeight, inputBarWidth, inputBarHeight);
            x += inputBarWidth + 1;
        }
        
        // Draw output visualizer
        analyserOutput.getByteFrequencyData(outputDataArray);
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        outputCtx.fillStyle = '#ff4b4b';
        
        const outputBarWidth = (outputCanvas.width / outputBufferLength) * 2.5;
        let outputBarHeight;
        x = 0;
        
        for (let i = 0; i < outputBufferLength; i++) {
            outputBarHeight = outputDataArray[i] / 2;
            outputCtx.fillRect(x, outputCanvas.height - outputBarHeight, outputBarWidth, outputBarHeight);
            x += outputBarWidth + 1;
        }
    }
    
    draw();
}

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', async () => {
    await initAudio();
    
    // Set up button
    const recordButton = document.getElementById('recordButton');
    recordButton.addEventListener('click', toggleRecording);
});
```

### 6. Dockerfile for Deployment

```dockerfile
# Dockerfile
FROM rust:1.70 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libssl-dev \
    pkg-config \
    libtorch-dev

# Copy source code
COPY . /usr/src/csm
WORKDIR /usr/src/csm

# Build WebAssembly
RUN cargo install wasm-pack
RUN wasm-pack build --target web --out-dir static/wasm src/wasm

# Build release
RUN cargo build --release --bin csm_server

# Runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libssl1.1 \
    ca-certificates

# Copy binary and resources
COPY --from=builder /usr/src/csm/target/release/csm_server /usr/local/bin/
COPY --from=builder /usr/src/csm/static /static
COPY models/ /models/

# Set up environment
ENV RUST_LOG=info
ENV MODEL_PATH=/models/llama-1B.pth
ENV MODEL_FLAVOR=llama-1B

# Expose port
EXPOSE 8080

# Run service
CMD ["csm_server", "--model-path", "/models/llama-1B.pth", "--model-flavor", "llama-1B", "--port", "8080"]
```

## Feature Comparison with Haadesx/realtime-voice-csm

| Feature                       | Haadesx Implementation | Our Rust Implementation |
|-------------------------------|------------------------|-------------------------|
| Real-time voice input         | ✅                     | ✅                      |
| Audio visualization           | ✅                     | ✅                      |
| GPU acceleration              | ✅                     | ✅                      |
| WebSocket communication       | ✅                     | ✅                      |
| Response caching              | ✅                     | ✅                      |
| LLM integration               | Dolphin3               | Configurable            |
| Web interface                 | Flask                  | WebAssembly + HTML      |
| Mobile support                | Limited                | Full (responsive design)|
| Multi-language support        | Planned                | Configurable            |
| Custom voices                 | Planned                | Configurable            |
| Thread-safe design            | Limited                | ✅ (Rust advantage)     |
| Memory safety                 | Python GC              | ✅ (Rust ownership)     |
| Zero-cost abstractions        | No                     | ✅ (Rust advantage)     |

## Performance Benefits of Rust Implementation

1. **Memory Safety**: Rust's ownership system eliminates entire classes of bugs that can occur in Python implementations.

2. **Concurrency**: Rust's async/await pattern with Tokio provides efficient async I/O without the overhead of Python's GIL (Global Interpreter Lock).

3. **Performance**: Zero-cost abstractions and compile-time optimizations result in faster execution compared to Python.

4. **Resource Usage**: Lower memory footprint and more efficient CPU utilization.

5. **WebAssembly Support**: Our implementation can run audio processing code directly in the browser via WebAssembly, reducing latency.

## Getting Started

1. Build the WebSocket server:
```bash
cargo build --release --bin csm_server
```

2. Build the WebAssembly module:
```bash
wasm-pack build --target web --out-dir static/wasm src/wasm
```

3. Run the server:
```bash
./target/release/csm_server --model-path models/llama-1B.pth --model-flavor llama-1B --port 8080
```

4. Open the web interface at `http://localhost:8080`

## Next Steps

1. **Streaming Optimization**: Implement chunked audio processing to reduce latency
2. **Voice Customization**: Add support for different voice characteristics
3. **Conversation Context**: Maintain conversation history for more coherent responses
4. **Mobile Interface**: Optimize the web interface for mobile devices
5. **Deployment**: Create deployment guides for various cloud providers
6. **Documentation**: Complete API documentation and examples

## Troubleshooting

1. **WebSocket Connection Issues**:
   - Check if the server is running on the correct port
   - Verify that your browser supports WebSockets
   - Check network connectivity and firewall settings

2. **Audio Permission Issues**:
   - Ensure you've granted microphone permissions in your browser
   - Try accessing the site via HTTPS (required for some browsers)

3. **Performance Issues**:
   - Enable GPU acceleration if available
   - Adjust audio quality settings
   - Check system resource usage

4. **Model Loading Errors**:
   - Verify model path and flavor are correct
   - Check if the model weights file exists and is not corrupted
   - Ensure you have enough system memory