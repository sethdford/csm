use tch::{Device, Kind, Tensor, nn, IndexOp};
use anyhow::{Result, Error};
use thiserror::Error;

use crate::models::llama::LlamaModel;

pub mod llama;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
    #[error("Model initialization failed: {0}")]
    InitializationError(String),
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone_flavor: String,
    pub decoder_flavor: String,
    pub text_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_num_codebooks: usize,
    pub max_seq_len: usize,
    pub embed_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backbone_flavor: "llama-1B".to_string(),
            decoder_flavor: "default".to_string(),
            text_vocab_size: 128_256,
            audio_vocab_size: 1024,
            audio_num_codebooks: 8,
            max_seq_len: 2048,
            embed_dim: 2048,
        }
    }
}

pub trait Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn generate(&self, input: &Tensor, max_length: usize) -> Result<Tensor>;
}

pub fn create_model(config: ModelConfig, device: Device) -> Result<Box<dyn Model>> {
    match config.backbone_flavor.as_str() {
        "llama-1B" => Ok(Box::new(LlamaModel::new(&nn::VarStore::new(device).root(), config)?)),
        _ => Err(ModelError::InvalidConfig(format!("Unknown backbone flavor: {}", config.backbone_flavor)).into()),
    }
}

pub fn sample_topk(logits: &Tensor, topk: i64, temperature: f32) -> Tensor {
    let mut probs = logits.softmax(-1, Kind::Float);
    if temperature != 1.0 {
        let log_probs = probs.log();
        let scaled_log_probs = log_probs / (temperature as f64);
        probs = scaled_log_probs.exp();
    }
    
    let (values, indices) = probs.topk(topk, -1, true, true);
    let dist = values.softmax(-1, Kind::Float);
    
    let sample = dist.multinomial(1, true);
    indices.gather(-1, &sample, false)
} 