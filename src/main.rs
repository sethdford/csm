use anyhow::Result;
use clap::Parser;
use tch::{Device, Kind, IndexOp};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use std::path::PathBuf;

mod models;
mod audio;
mod utils;

use audio::AudioProcessor;
use models::{Model, ModelConfig};
use utils::prepare_audio_input;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model weights
    #[arg(short = 'w', long)]
    model_path: PathBuf,

    /// Model flavor (e.g., llama-1B)
    #[arg(short = 'f', long)]
    model_flavor: String,

    /// Input audio file
    #[arg(short = 'i', long)]
    input_file: PathBuf,

    /// Output audio file
    #[arg(short = 'o', long)]
    output_file: PathBuf,

    /// Maximum number of new tokens to generate
    #[arg(short = 'm', long, default_value_t = 100)]
    max_new_tokens: usize,

    /// Temperature for sampling
    #[arg(short = 't', long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-k for sampling
    #[arg(short = 'k', long, default_value_t = 50)]
    top_k: usize,

    /// Top-p for sampling
    #[arg(short = 'p', long, default_value_t = 0.9)]
    top_p: f64,
}

fn main() -> Result<()> {
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
    tracing::info!("Loading model weights from {}", args.model_path.display());
    vs.load(&args.model_path)?;
    tracing::info!("Model weights loaded successfully");

    // Initialize audio processor
    let audio_processor = AudioProcessor::new(44100, 1, 16);

    // Load and process input audio
    info!("Processing input audio: {}", args.input_file.display());
    let input_audio = audio_processor.read_wav(&args.input_file)?;
    let input_tokens = audio_processor.tokenize(&input_audio, config.audio_num_codebooks)?;

    // Convert tokens to tensor
    let input_tensor = tch::Tensor::zeros(&[1, input_tokens[0].len() as i64], (Kind::Int64, device));
    for (i, token) in input_tokens[0].iter().enumerate() {
        input_tensor.i((0, i as i64)).copy_(&tch::Tensor::from(*token));
    }

    // Generate output
    let output_tokens = model.generate(&input_tensor, args.max_new_tokens)?;

    // Convert output tokens to audio
    let mut output_tokens_vec = vec![0i64; output_tokens.numel() as usize];
    output_tokens.copy_data(&mut output_tokens_vec, output_tokens.numel() as usize);
    let output_tokens_vec = vec![output_tokens_vec];

    // Convert tokens back to audio
    let output_audio = audio_processor.detokenize(&output_tokens_vec, output_tokens.numel() as usize)?;

    // Save output audio
    info!("Saving output to {}", args.output_file.display());
    audio_processor.write_wav(&args.output_file, &output_audio)?;

    Ok(())
} 