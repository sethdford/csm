use csm::models::{self, Model, ModelConfig};
use csm::server::websocket::WebSocketServer;
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tch::Device;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

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
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();
    
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