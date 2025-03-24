use anyhow::Result;
use csm::audio::AudioProcessor;
use ndarray::Array2;

fn main() -> Result<()> {
    // Create a simple sine wave
    let sample_rate = 44100;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note
    let num_samples = (sample_rate as f32 * duration) as usize;
    
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(sample);
    }

    // Create mono audio data
    let audio = Array2::from_shape_vec((num_samples, 1), samples)?;

    // Initialize audio processor
    let processor = AudioProcessor::default();

    // Write the audio to a WAV file
    processor.write_wav("test_sine.wav", &audio)?;

    // Read it back
    let read_audio = processor.read_wav("test_sine.wav")?;

    // Verify the data
    assert_eq!(read_audio.shape(), audio.shape());
    
    println!("Successfully wrote and read back audio data!");
    println!("Audio shape: {:?}", audio.shape());
    println!("Sample rate: {} Hz", sample_rate);
    println!("Duration: {} seconds", duration);
    println!("Frequency: {} Hz", frequency);

    Ok(())
} 