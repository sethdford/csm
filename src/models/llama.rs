use anyhow::{anyhow, Result};
use tch::{nn, Device, Kind, Tensor};
use tch::nn::{Module, Path};
use std::path::PathBuf;

use super::{Model, ModelConfig, ModelError};

pub struct LlamaConfig {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub embed_dim: usize,
    pub max_seq_len: usize,
    pub intermediate_dim: usize,
    pub attn_dropout: f64,
    pub norm_eps: f64,
    pub rope_base: usize,
    pub scale_factor: usize,
}

impl LlamaConfig {
    pub fn llama_1b() -> Self {
        Self {
            vocab_size: 128_256,
            num_layers: 16,
            num_heads: 32,
            num_kv_heads: 8,
            embed_dim: 2048,
            max_seq_len: 2048,
            intermediate_dim: 8192,
            attn_dropout: 0.0,
            norm_eps: 1e-5,
            rope_base: 500_000,
            scale_factor: 32,
        }
    }

    pub fn llama_100m() -> Self {
        Self {
            vocab_size: 128_256,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 2,
            embed_dim: 1024,
            max_seq_len: 2048,
            intermediate_dim: 8192,
            attn_dropout: 0.0,
            norm_eps: 1e-5,
            rope_base: 500_000,
            scale_factor: 32,
        }
    }
}

pub struct LlamaModel {
    token_embedding: nn::Embedding,
    layers: Vec<TransformerLayer>,
    norm: nn::LayerNorm,
    output: nn::Linear,
}

struct TransformerLayer {
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    self_attn: SelfAttention,
    mlp: MLP,
}

struct SelfAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
}

struct MLP {
    w1: nn::Linear,
    w2: nn::Linear,
}

impl MLP {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.w1.forward(x);
        let hidden = hidden.gelu("none");
        let output = self.w2.forward(&hidden);
        Ok(output)
    }
}

impl LlamaModel {
    pub fn new(vs: &nn::Path, config: ModelConfig) -> Result<Self> {
        let token_embedding = nn::embedding(
            vs,
            config.text_vocab_size as i64,
            config.embed_dim as i64,
            Default::default(),
        );

        let mut layers = Vec::new();
        for i in 0..12 {
            let layer_vs = vs.sub(&format!("layer_{}", i));
            layers.push(TransformerLayer {
                norm1: nn::layer_norm(
                    &layer_vs,
                    vec![config.embed_dim as i64],
                    Default::default(),
                ),
                norm2: nn::layer_norm(
                    &layer_vs,
                    vec![config.embed_dim as i64],
                    Default::default(),
                ),
                self_attn: SelfAttention {
                    q_proj: nn::linear(
                        &layer_vs,
                        config.embed_dim as i64,
                        config.embed_dim as i64,
                        Default::default(),
                    ),
                    k_proj: nn::linear(
                        &layer_vs,
                        config.embed_dim as i64,
                        config.embed_dim as i64,
                        Default::default(),
                    ),
                    v_proj: nn::linear(
                        &layer_vs,
                        config.embed_dim as i64,
                        config.embed_dim as i64,
                        Default::default(),
                    ),
                    o_proj: nn::linear(
                        &layer_vs,
                        config.embed_dim as i64,
                        config.embed_dim as i64,
                        Default::default(),
                    ),
                },
                mlp: MLP {
                    w1: nn::linear(
                        &layer_vs,
                        config.embed_dim as i64,
                        4096,
                        Default::default(),
                    ),
                    w2: nn::linear(
                        &layer_vs,
                        4096,
                        config.embed_dim as i64,
                        Default::default(),
                    ),
                },
            });
        }

        let norm = nn::layer_norm(
            vs,
            vec![config.embed_dim as i64],
            Default::default(),
        );

        let output = nn::linear(
            vs,
            config.embed_dim as i64,
            config.audio_vocab_size as i64,
            Default::default(),
        );

        Ok(Self {
            token_embedding,
            layers,
            norm,
            output,
        })
    }
}

impl Model for LlamaModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_states = self.token_embedding.forward(input);
        let mut x = hidden_states;

        for layer in &self.layers {
            let normed = layer.norm1.forward(&x);
            let q = layer.self_attn.q_proj.forward(&normed);
            let k = layer.self_attn.k_proj.forward(&normed);
            let v = layer.self_attn.v_proj.forward(&normed);

            // Get dimensions
            let (batch_size, seq_len, hidden_size) = q.size3().unwrap();
            
            // Reshape q, k, v for attention
            let q = q.reshape(&[batch_size, seq_len, hidden_size]);
            let k = k.reshape(&[batch_size, seq_len, hidden_size]);
            let v = v.reshape(&[batch_size, seq_len, hidden_size]);

            let attn_weights = q.matmul(&k.transpose(-2, -1));
            let scale = (hidden_size as f64).sqrt();
            let attn_weights = attn_weights / scale;
            let attn_weights = attn_weights.softmax(-1, Kind::Float);

            let attn_output = attn_weights.matmul(&v);
            let attn_output = layer.self_attn.o_proj.forward(&attn_output);

            x = x + attn_output;
            let normed = layer.norm2.forward(&x);
            let mlp_output = layer.mlp.forward(&normed)?;
            x = x + mlp_output;
        }

        let x = self.norm.forward(&x);
        let logits = self.output.forward(&x);
        Ok(logits)
    }

    fn generate(&self, input: &Tensor, max_length: usize) -> Result<Tensor> {
        let mut current_input = input.shallow_clone();
        let mut generated = Vec::new();

        for _ in 0..max_length {
            let output = self.forward(&current_input)?;
            let next_token = output.select(1, -1).argmax(-1, false);
            generated.push(next_token.copy());
            
            current_input = Tensor::cat(&[current_input, next_token.view([1, 1])], 1);
        }

        Ok(Tensor::stack(&generated, 0))
    }
}