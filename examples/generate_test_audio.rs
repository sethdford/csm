use anyhow::Result;
use csm::audio::AudioProcessor;
use ndarray::Array2;

fn generate_tone(sample_rate: u32, duration: f32, frequency: f32) -> Array2<f32> {
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Add some amplitude variation to make it more "speech-like"
        let envelope = 0.5 + 0.3 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
        let sample = envelope * (2.0 * std::f32::consts::PI * frequency * t).sin();
        samples.push(sample);
    }
    
    Array2::from_shape_vec((num_samples, 1), samples).unwrap()
}

fn main() -> Result<()> {
    let processor = AudioProcessor::new(44100, 1, 16);
    
    // Generate two different "conversational" audio clips
    // First clip: lower frequency, shorter duration
    let audio_a = generate_tone(44100, 1.5, 150.0);
    processor.write_wav("audio/conversational_a.wav", &audio_a)?;
    println!("Generated conversational_a.wav");
    
    // Second clip: higher frequency, longer duration
    let audio_b = generate_tone(44100, 2.0, 200.0);
    processor.write_wav("audio/conversational_b.wav", &audio_b)?;
    println!("Generated conversational_b.wav");
    
    Ok(())
} 