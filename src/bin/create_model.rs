use anyhow::Result;
use tch::{nn, Device, Kind, Tensor};
use csm::models::{Model, ModelConfig};

fn main() -> Result<()> {
    let config = ModelConfig {
        backbone_flavor: "llama-1B".to_string(),
        decoder_flavor: "default".to_string(),
        text_vocab_size: 128_256,
        audio_vocab_size: 1024,
        audio_num_codebooks: 8,
        max_seq_len: 2048,
        embed_dim: 2048,
    };

    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let root = vs.root();

    // Initialize token embeddings
    root.var_copy(
        "token_embedding_weight",
        &(&Tensor::randn(&[config.text_vocab_size as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
    );

    // Initialize transformer layers
    for i in 0..12 { // Fixed number of layers for llama-1B
        let layer_root = root.sub(&format!("layer_{}", i));
        
        // Layer norms
        layer_root.var_copy(
            "norm1_weight",
            &Tensor::ones(&[config.embed_dim as i64], (Kind::Float, device))
        );
        layer_root.var_copy(
            "norm1_bias",
            &Tensor::zeros(&[config.embed_dim as i64], (Kind::Float, device))
        );
        layer_root.var_copy(
            "norm2_weight",
            &Tensor::ones(&[config.embed_dim as i64], (Kind::Float, device))
        );
        layer_root.var_copy(
            "norm2_bias",
            &Tensor::zeros(&[config.embed_dim as i64], (Kind::Float, device))
        );

        // Self attention
        layer_root.var_copy(
            "self_attn_q_proj_weight",
            &(&Tensor::randn(&[config.embed_dim as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
        );
        layer_root.var_copy(
            "self_attn_k_proj_weight",
            &(&Tensor::randn(&[config.embed_dim as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
        );
        layer_root.var_copy(
            "self_attn_v_proj_weight",
            &(&Tensor::randn(&[config.embed_dim as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
        );
        layer_root.var_copy(
            "self_attn_o_proj_weight",
            &(&Tensor::randn(&[config.embed_dim as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
        );

        // MLP
        layer_root.var_copy(
            "mlp_w1_weight",
            &(&Tensor::randn(&[config.embed_dim as i64, 4096], (Kind::Float, device)) * 0.02)
        );
        layer_root.var_copy(
            "mlp_w2_weight",
            &(&Tensor::randn(&[4096, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
        );
    }

    // Final layer norm
    root.var_copy(
        "norm_weight",
        &Tensor::ones(&[config.embed_dim as i64], (Kind::Float, device))
    );
    root.var_copy(
        "norm_bias",
        &Tensor::zeros(&[config.embed_dim as i64], (Kind::Float, device))
    );

    // Output projection
    root.var_copy(
        "output_weight",
        &(&Tensor::randn(&[config.audio_vocab_size as i64, config.embed_dim as i64], (Kind::Float, device)) * 0.02)
    );
    root.var_copy(
        "output_bias",
        &Tensor::zeros(&[config.audio_vocab_size as i64], (Kind::Float, device))
    );

    // Save the model
    vs.save("models/llama-1B.pth")?;
    println!("Model weights saved to models/llama-1B.pth");
    Ok(())
} 