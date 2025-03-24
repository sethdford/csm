use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavReader, WavWriter};
use ndarray::Array2;
use std::path::Path;

pub struct AudioProcessor {
    sample_rate: u32,
    num_channels: u16,
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            num_channels: 1,
        }
    }
}

impl AudioProcessor {
    pub fn new(sample_rate: u32, num_channels: u16, _bits_per_sample: u16) -> Self {
        Self {
            sample_rate,
            num_channels,
        }
    }

    pub fn read_wav<P: AsRef<Path>>(&self, path: P) -> Result<Array2<f32>> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        if spec.channels != self.num_channels {
            return Err(anyhow!(
                "Expected {} channels, got {}",
                self.num_channels,
                spec.channels
            ));
        }

        let samples: Vec<f32> = match spec.sample_format {
            SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
            SampleFormat::Int => reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / i32::MAX as f32))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let num_frames = samples.len() / spec.channels as usize;
        Ok(Array2::from_shape_vec(
            (num_frames, spec.channels as usize),
            samples,
        )?)
    }

    pub fn write_wav<P: AsRef<Path>>(&self, path: P, audio: &Array2<f32>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.num_channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer = WavWriter::create(path, spec)?;

        for sample in audio.iter() {
            writer.write_sample(*sample)?;
        }

        Ok(())
    }

    pub fn tokenize(&self, audio: &Array2<f32>, num_codebooks: usize) -> Result<Vec<Vec<i64>>> {
        // Simple quantization-based tokenization
        // In practice, you'd want to use a more sophisticated method like VQ-VAE
        let mut tokens = Vec::with_capacity(num_codebooks);
        
        for _ in 0..num_codebooks {
            let mut codebook_tokens = Vec::with_capacity(audio.nrows());
            
            for frame in audio.rows() {
                // Simple scalar quantization - just an example
                // You'd want to replace this with proper codebook lookup
                let token = ((frame[0] + 1.0) * 512.0) as i64;
                let token = token.clamp(0, 1023); // Clamp to 10-bit range
                codebook_tokens.push(token);
            }
            
            tokens.push(codebook_tokens);
        }

        Ok(tokens)
    }

    pub fn detokenize(&self, tokens: &[Vec<i64>], num_frames: usize) -> Result<Array2<f32>> {
        let num_codebooks = tokens.len();
        let mut audio = Array2::zeros((num_frames, self.num_channels as usize));

        // Simple inverse quantization
        // In practice, you'd want to use a proper codebook-based reconstruction
        for frame_idx in 0..num_frames {
            let mut sample = 0.0;
            for codebook_idx in 0..num_codebooks {
                let token = tokens[codebook_idx][frame_idx];
                sample += (token as f32 / 512.0) - 1.0;
            }
            sample /= num_codebooks as f32;
            
            for channel in 0..self.num_channels as usize {
                audio[[frame_idx, channel]] = sample;
            }
        }

        Ok(audio)
    }
} 