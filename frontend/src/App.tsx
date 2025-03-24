import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface AudioState {
  isRecording: boolean;
  isPlaying: boolean;
  error: string | null;
}

const WS_ENDPOINT = process.env.REACT_APP_WS_ENDPOINT || 'wss://api.yourdomain.com/ws';

function App() {
  const [audioState, setAudioState] = useState<AudioState>({
    isRecording: false,
    isPlaying: false,
    error: null,
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    wsRef.current = new WebSocket(WS_ENDPOINT);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = async (event) => {
      const response = JSON.parse(event.data);
      if (response.type === 'audio') {
        await playAudioResponse(response.data);
      }
    };
    
    wsRef.current.onerror = (error) => {
      setAudioState(prev => ({ ...prev, error: 'WebSocket error occurred' }));
    };
    
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            action: 'audio',
            data: e.data
          }));
        }
      };
      
      mediaRecorderRef.current.start(100);
      setAudioState(prev => ({ ...prev, isRecording: true }));
    } catch (error) {
      setAudioState(prev => ({ 
        ...prev, 
        error: 'Error accessing microphone' 
      }));
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setAudioState(prev => ({ ...prev, isRecording: false }));
  };

  const playAudioResponse = async (audioData: string) => {
    try {
      setAudioState(prev => ({ ...prev, isPlaying: true }));
      
      const arrayBuffer = Uint8Array.from(atob(audioData), c => c.charCodeAt(0));
      const audioContext = audioContextRef.current || new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.buffer);
      
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      
      source.onended = () => {
        setAudioState(prev => ({ ...prev, isPlaying: false }));
      };
      
      source.start(0);
    } catch (error) {
      setAudioState(prev => ({ 
        ...prev, 
        isPlaying: false,
        error: 'Error playing audio response' 
      }));
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CSM Voice Demo</h1>
        <p>Experience natural voice interactions powered by CSM</p>
      </header>
      
      <main className="App-main">
        <div className="conversation-container">
          <button
            className={`record-button ${audioState.isRecording ? 'recording' : ''}`}
            onClick={audioState.isRecording ? stopRecording : startRecording}
            disabled={audioState.isPlaying}
          >
            {audioState.isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
          
          {audioState.isPlaying && (
            <div className="playing-indicator">
              Playing response...
            </div>
          )}
          
          {audioState.error && (
            <div className="error-message">
              {audioState.error}
              <button 
                onClick={() => setAudioState(prev => ({ ...prev, error: null }))}
              >
                Dismiss
              </button>
            </div>
          )}
        </div>
      </main>
      
      <footer className="App-footer">
        <p>Powered by CSM - A Rust implementation inspired by Sesame</p>
      </footer>
    </div>
  );
}

export default App; 