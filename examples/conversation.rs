use anyhow::Result;
use csm::audio::AudioProcessor;
use ndarray::Array2;

fn main() -> Result<()> {
    println!("Testing audio processing functionality...");

    // Initialize audio processor
    let audio_processor = AudioProcessor::new(44100, 1, 16);

    // Read the first conversation audio file
    println!("Reading conversational_a.wav...");
    let audio_a = audio_processor.read_wav("audio/conversational_a.wav")?;
    println!("Audio A loaded: {} frames", audio_a.nrows());

    // Read the second conversation audio file
    println!("Reading conversational_b.wav...");
    let audio_b = audio_processor.read_wav("audio/conversational_b.wav")?;
    println!("Audio B loaded: {} frames", audio_b.nrows());

    // Concatenate the audio files
    let total_frames = audio_a.nrows() + audio_b.nrows();
    let mut combined_audio = Array2::zeros((total_frames, 1));
    
    combined_audio.slice_mut(ndarray::s![..audio_a.nrows(), ..])
        .assign(&audio_a);
    combined_audio.slice_mut(ndarray::s![audio_a.nrows().., ..])
        .assign(&audio_b);

    // Save the combined audio
    println!("Saving combined audio to combined_conversation.wav...");
    audio_processor.write_wav("combined_conversation.wav", &combined_audio)?;
    println!("Successfully saved combined_conversation.wav");

    Ok(())
} 