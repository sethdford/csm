use crate::audio::AudioProcessor;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{self, Receiver, Sender};

/// Handles streaming audio processing with async communication channels
pub struct AudioStream {
    processor: Arc<Mutex<AudioProcessor>>,
    input_sender: Sender<Vec<f32>>,
    output_receiver: Receiver<Vec<f32>>,
    sample_rate: usize,
    channels: usize,
}

impl AudioStream {
    /// Creates a new audio stream with the specified parameters
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
                if let Ok(tokens) = processor.tokenize_chunk(&audio_chunk) {
                    // TODO: Process tokens with model
                    
                    // For now, just echo the input
                    if let Ok(output_audio) = processor.detokenize_chunk(&tokens) {
                        let _ = output_sender.send(output_audio).await;
                    }
                }
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
    
    /// Sends an audio chunk for processing
    pub async fn process_audio_chunk(&self, chunk: Vec<f32>) -> Result<()> {
        Ok(self.input_sender.send(chunk).await?)
    }
    
    /// Receives the next processed audio chunk
    pub async fn get_next_output(&mut self) -> Option<Vec<f32>> {
        self.output_receiver.recv().await
    }
    
    /// Converts a raw byte array to audio samples
    pub fn bytes_to_samples(&self, bytes: &[u8]) -> Vec<f32> {
        let mut samples = Vec::with_capacity(bytes.len() / 4);
        
        // Convert bytes to f32 samples (assuming 32-bit float format)
        let mut i = 0;
        while i < bytes.len() {
            if i + 4 <= bytes.len() {
                let sample_bytes = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
                let sample = f32::from_le_bytes(sample_bytes);
                samples.push(sample);
            }
            i += 4;
        }
        
        samples
    }
    
    /// Converts audio samples to a byte array
    pub fn samples_to_bytes(&self, samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        
        for sample in samples {
            let sample_bytes = sample.to_le_bytes();
            bytes.extend_from_slice(&sample_bytes);
        }
        
        bytes
    }
}

/// Extension trait for AudioProcessor to handle chunked audio
pub trait ChunkedAudioProcessing {
    /// Tokenizes a chunk of audio samples
    fn tokenize_chunk(&self, chunk: &[f32]) -> Result<Vec<Vec<i64>>>;
    
    /// Detokenizes tokens into audio samples
    fn detokenize_chunk(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>>;
}

impl ChunkedAudioProcessing for AudioProcessor {
    fn tokenize_chunk(&self, chunk: &[f32]) -> Result<Vec<Vec<i64>>> {
        // This is a simplified implementation
        // In a real implementation, this would properly tokenize audio
        let mut tokens = Vec::new();
        let codebooks = 8; // Matching audio_num_codebooks in config
        
        // Create a simple token representation (placeholder)
        for _ in 0..codebooks {
            tokens.push(vec![0i64; chunk.len() / 1024 + 1]);
        }
        
        Ok(tokens)
    }
    
    fn detokenize_chunk(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        // This is a simplified implementation
        // In a real implementation, this would properly detokenize to audio
        let output_len = if tokens.is_empty() { 0 } else { tokens[0].len() * 1024 };
        let mut output = vec![0.0f32; output_len];
        
        // For now, just return silence
        Ok(output)
    }
} 