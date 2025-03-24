use anyhow::Result;
use tch::{Device, IndexOp, Kind, Tensor};

/// Convert a slice of i64 values to a tensor
pub fn to_tensor<T: AsRef<[i64]>>(values: T, device: Device) -> Tensor {
    Tensor::from_slice(values.as_ref()).to(device)
}

/// Create a position tensor for the given sequence length
pub fn create_position_ids(seq_len: i64, device: Device) -> Tensor {
    Tensor::arange(seq_len, (Kind::Int64, device))
}

/// Create an attention mask tensor
pub fn create_attention_mask(seq_len: i64, device: Device) -> Tensor {
    let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device));
    mask.triu(1).neg() * f64::INFINITY
}

/// Create a padding mask tensor
pub fn create_padding_mask(lengths: &[i64], max_len: i64, device: Device) -> Tensor {
    let batch_size = lengths.len() as i64;
    let mut mask = Tensor::ones(&[batch_size, max_len], (Kind::Bool, device));
    
    for (i, &length) in lengths.iter().enumerate() {
        if length < max_len {
            let zeros = Tensor::zeros(&[max_len - length], (Kind::Bool, device));
            mask.slice(0, i as i64, (i + 1) as i64, 1)
                .slice(1, length, max_len, 1)
                .copy_(&zeros);
        }
    }
    
    mask
}

/// Convert raw audio samples to model input tensors
pub fn prepare_audio_input(
    tokens: Vec<Vec<i64>>,
    device: Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = tokens.len();
    let max_length = tokens.iter().map(|seq| seq.len()).max().unwrap_or(0);

    // Create input tensor
    let mut input_tensor = Tensor::zeros(&[batch_size as i64, max_length as i64], (Kind::Int64, device));
    for (i, seq) in tokens.iter().enumerate() {
        let seq_tensor = Tensor::from_slice(seq).to_device(device);
        input_tensor.i((i as i64, ..seq.len() as i64)).copy_(&seq_tensor);
    }

    // Create attention mask
    let mut mask = Tensor::ones(&[batch_size as i64, max_length as i64], (Kind::Bool, device));
    for (i, seq) in tokens.iter().enumerate() {
        if seq.len() < max_length {
            mask.i((i as i64, seq.len() as i64..)).fill_(0);
        }
    }

    // Create position IDs
    let position_ids = Tensor::arange(max_length as i64, (Kind::Int64, device))
        .unsqueeze(0)
        .expand(&[batch_size as i64, max_length as i64], true);

    Ok((input_tensor, mask, position_ids))
}

/// Helper function to move tensors to the specified device
pub fn to_device(tensor: Tensor, device: Device) -> Tensor {
    if tensor.device() != device {
        tensor.to(device)
    } else {
        tensor
    }
} 