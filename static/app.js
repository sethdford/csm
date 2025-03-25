// Global variables
let isRecording = false;
let isConnected = false;
let websocket = null;
let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let analyserInput = null;
let analyserOutput = null;
let audioOutput = null;
let canvasInput = null;
let canvasOutput = null;
let ctxInput = null;
let ctxOutput = null;

// Initialize on load
document.addEventListener('DOMContentLoaded', async () => {
    await initAudio();
    setupEventListeners();
    setupVisualizers();
    connectWebSocket();
});

// Initialize AudioContext
async function initAudio() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Create analysers for visualization
        analyserInput = audioContext.createAnalyser();
        analyserInput.fftSize = 256;
        
        analyserOutput = audioContext.createAnalyser();
        analyserOutput.fftSize = 256;
        
        // Create audio output node
        audioOutput = audioContext.createGain();
        audioOutput.connect(analyserOutput);
        analyserOutput.connect(audioContext.destination);
        
        updateStatus('Ready');
    } catch (error) {
        console.error('Audio initialization error:', error);
        updateStatus('Audio system error', true);
    }
}

// Set up event listeners
function setupEventListeners() {
    const recordButton = document.getElementById('recordButton');
    recordButton.addEventListener('click', toggleRecording);
}

// Set up audio visualizers
function setupVisualizers() {
    canvasInput = document.getElementById('inputViz');
    canvasOutput = document.getElementById('outputViz');
    
    ctxInput = canvasInput.getContext('2d');
    ctxOutput = canvasOutput.getContext('2d');
    
    // Size canvases appropriately
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
    
    // Start visualization loop
    visualize();
}

// Resize canvas elements
function resizeCanvases() {
    const resizeCanvas = (canvas) => {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
    };
    
    resizeCanvas(canvasInput);
    resizeCanvas(canvasOutput);
}

// Connect to WebSocket server
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/audio`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
        isConnected = true;
        updateStatus('Connected');
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        isConnected = false;
        updateStatus('Disconnected', true);
        
        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection error', true);
    };
    
    websocket.onmessage = handleWebSocketMessage;
}

// Handle incoming WebSocket messages
async function handleWebSocketMessage(event) {
    if (event.data instanceof Blob) {
        try {
            const arrayBuffer = await event.data.arrayBuffer();
            const audioData = new Float32Array(arrayBuffer);
            
            // Play audio response
            const audioBuffer = audioContext.createBuffer(1, audioData.length, audioContext.sampleRate);
            audioBuffer.getChannelData(0).set(audioData);
            
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioOutput);
            source.start();
            
            updateStatus('Playing response');
            
            // When playback ends
            source.onended = () => {
                updateStatus('Ready');
            };
        } catch (error) {
            console.error('Error processing audio response:', error);
            updateStatus('Error processing response', true);
        }
    }
}

// Toggle recording state
async function toggleRecording() {
    if (!isConnected) {
        updateStatus('Not connected to server', true);
        return;
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

// Start audio recording
async function startRecording() {
    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Connect microphone to analyser for visualization
        const micSource = audioContext.createMediaStreamSource(mediaStream);
        micSource.connect(analyserInput);
        
        // Create media recorder
        mediaRecorder = new MediaRecorder(mediaStream);
        audioChunks = [];
        
        // Setup media recorder event handlers
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                
                // Send audio chunk to server
                if (isConnected) {
                    event.data.arrayBuffer().then(buffer => {
                        const float32Array = new Float32Array(buffer);
                        websocket.send(float32Array);
                    });
                }
            }
        };
        
        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;
        
        // Update UI
        document.getElementById('recordButton').classList.add('recording');
        document.getElementById('recordButton').innerHTML = '<span>Release to Stop</span>';
        updateStatus('Listening...');
    } catch (error) {
        console.error('Error starting recording:', error);
        updateStatus('Microphone access error', true);
    }
}

// Stop audio recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        
        // Release microphone
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        
        // Update UI
        document.getElementById('recordButton').classList.remove('recording');
        document.getElementById('recordButton').innerHTML = '<span>Press to Talk</span>';
        updateStatus('Processing...');
    }
}

// Draw audio visualizers
function visualize() {
    if (!analyserInput || !analyserOutput || !ctxInput || !ctxOutput) {
        requestAnimationFrame(visualize);
        return;
    }
    
    // Input visualizer
    const inputBufferLength = analyserInput.frequencyBinCount;
    const inputDataArray = new Uint8Array(inputBufferLength);
    analyserInput.getByteFrequencyData(inputDataArray);
    
    // Output visualizer
    const outputBufferLength = analyserOutput.frequencyBinCount;
    const outputDataArray = new Uint8Array(outputBufferLength);
    analyserOutput.getByteFrequencyData(outputDataArray);
    
    // Clear canvases
    ctxInput.clearRect(0, 0, canvasInput.width, canvasInput.height);
    ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    
    // Draw input visualizer
    ctxInput.fillStyle = '#00c6ff';
    drawBars(ctxInput, inputDataArray, canvasInput.width, canvasInput.height);
    
    // Draw output visualizer
    ctxOutput.fillStyle = '#ff4b4b';
    drawBars(ctxOutput, outputDataArray, canvasOutput.width, canvasOutput.height);
    
    // Continue animation
    requestAnimationFrame(visualize);
}

// Draw frequency bars
function drawBars(ctx, dataArray, width, height) {
    const barWidth = (width / dataArray.length) * 2.5;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * height;
        
        ctx.fillRect(x, height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
    }
}

// Update status message
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    
    if (isError) {
        statusElement.classList.add('error');
    } else {
        statusElement.classList.remove('error');
    }
} 