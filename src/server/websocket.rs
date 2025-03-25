use crate::audio::streaming::{AudioStream, ChunkedAudioProcessing};
use crate::models::{Model, ModelConfig};
use anyhow::Result;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use futures::{FutureExt, StreamExt};
use tracing::{info, error, debug};
use warp::{ws::{WebSocket, Message}, Filter};

/// WebSocket server for real-time audio processing
pub struct WebSocketServer {
    audio_stream: Arc<Mutex<AudioStream>>,
    model: Arc<Mutex<dyn Model + Send>>,
}

impl WebSocketServer {
    /// Creates a new WebSocket server with the given model
    pub fn new(model: Arc<Mutex<dyn Model + Send>>, _config: &ModelConfig) -> Self {
        let audio_stream = Arc::new(Mutex::new(
            AudioStream::new(
                44100,  // Sample rate
                1,      // Mono audio
                16      // Bit depth
            )
        ));
        
        Self {
            audio_stream,
            model,
        }
    }
    
    /// Starts the WebSocket server on the specified port
    pub async fn start(&self, port: u16) -> Result<()> {
        let audio_stream = Arc::clone(&self.audio_stream);
        let model = Arc::clone(&self.model);
        
        // Serve static files
        let static_files = warp::path("static")
            .and(warp::fs::dir("static"));
        
        // Serve index.html for root path
        let index = warp::path::end()
            .and(warp::fs::file("static/index.html"));
        
        // Create WebSocket route
        let ws_route = warp::path("audio")
            .and(warp::ws())
            .and(with_audio_stream(audio_stream))
            .and(with_model(model))
            .map(|ws: warp::ws::Ws, audio_stream, model| {
                ws.on_upgrade(move |socket| handle_connection(socket, audio_stream, model))
            });
            
        // Combine routes
        let routes = ws_route
            .or(static_files)
            .or(index)
            .with(warp::cors().allow_any_origin());
        
        // Start the server
        info!("Starting WebSocket server on port {}", port);
        warp::serve(routes)
            .run(([0, 0, 0, 0], port))
            .await;
            
        Ok(())
    }
}

/// Helper function to include AudioStream in route handlers
fn with_audio_stream(
    audio_stream: Arc<Mutex<AudioStream>>,
) -> impl Filter<Extract = (Arc<Mutex<AudioStream>>,), Error = Infallible> + Clone {
    warp::any().map(move || Arc::clone(&audio_stream))
}

/// Helper function to include Model in route handlers
fn with_model(
    model: Arc<Mutex<dyn Model + Send>>,
) -> impl Filter<Extract = (Arc<Mutex<dyn Model + Send>>,), Error = Infallible> + Clone {
    warp::any().map(move || Arc::clone(&model))
}

/// Handles WebSocket connections
async fn handle_connection(
    ws: WebSocket, 
    audio_stream: Arc<Mutex<AudioStream>>, 
    _model: Arc<Mutex<dyn Model + Send>>
) {
    let (mut ws_sender, mut ws_receiver) = ws.split();
    let (tx, mut rx) = mpsc::channel::<Message>(100);
    
    info!("New WebSocket connection established");
    
    // Handle incoming audio chunks
    let audio_stream_clone = Arc::clone(&audio_stream);
    let incoming = async move {
        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(msg) => {
                    if msg.is_binary() {
                        let bytes = msg.as_bytes();
                        debug!("Received audio chunk: {} bytes", bytes.len());
                        
                        // Convert bytes to audio samples
                        let stream_guard = audio_stream_clone.lock().unwrap();
                        let audio_chunk = stream_guard.bytes_to_samples(bytes);
                        drop(stream_guard); // Release lock before async operation
                        
                        // Process audio chunk
                        let mut stream_guard = audio_stream_clone.lock().unwrap();
                        if let Err(e) = stream_guard.process_audio_chunk(audio_chunk).await {
                            error!("Error processing audio chunk: {}", e);
                            continue;
                        }
                        
                        // Get output and send back
                        if let Some(output) = stream_guard.get_next_output().await {
                            let output_bytes = stream_guard.samples_to_bytes(&output);
                            debug!("Sending response: {} bytes", output_bytes.len());
                            let _ = tx.send(Message::binary(output_bytes)).await;
                        }
                    } else if msg.is_close() {
                        info!("WebSocket connection closed by client");
                        break;
                    }
                },
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }
        
        info!("WebSocket connection handler completed");
    };
    
    // Send responses back to client
    let outgoing = async move {
        while let Some(msg) = rx.recv().await {
            if let Err(e) = ws_sender.send(msg).await {
                error!("Error sending WebSocket message: {}", e);
                break;
            }
        }
    };
    
    // Run both tasks concurrently
    tokio::select! {
        _ = incoming => {},
        _ = outgoing => {},
    }
} 