

# Soni Voice Assistant

An AI-powered voice assistant project developed for the SoICT Innovation Club (Sinno) at Hanoi University of Science and Technology. This voice assistant provides intelligent, conversational interactions in Vietnamese, designed to introduce and represent the club.

## Features

- **Vietnamese Language Support**: Native Vietnamese ASR (Automatic Speech Recognition) and TTS (Text-to-Speech)
- **Voice Interaction**: Real-time speech recognition and text-to-speech synthesis
- **Multiple TTS Voices**: Support for GLaDOS, Kokoro, and Vietnamese Piper voices
- **Advanced ASR**: Zipformer-based Vietnamese speech recognition optimized for Vietnamese language
- **Voice Activity Detection (VAD)**: Automatic speech detection using Silero VAD
- **Customizable Personality**: Configure custom personality prompts and announcements
- **Interruptible Mode**: Option to interrupt ongoing speech
- **Edge Device Optimized**: Runs smoothly on Raspberry Pi 4B with 4GB RAM and other edge devices

## Prerequisites

- **Operating System**: Linux (Ubuntu/Debian recommended, Raspberry Pi OS supported)
- **Hardware**: Raspberry Pi 4B (4GB RAM) or equivalent edge device
- **Python**: >= 3.12
- **Audio Hardware**: Working microphone and speakers
- **Dependencies**: PortAudio library

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tourmii/soni-voice-assistant.git
cd soni-voice-assistant
```

### 2. Install System Dependencies

Install PortAudio library (required for audio I/O):

```bash
sudo apt update
sudo apt install libportaudio2
```

### 3. Install Python Dependencies

The project uses `uv` for dependency management. If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

Or use the installation script:

```bash
python scripts/install.py
```

### 4. Download Model Files

Download all required AI models (ASR, TTS, VAD):

```bash
uv run glados download
```

This will download:
- Zipformer Vietnamese ASR models for Vietnamese speech recognition (~200MB)
- Silero VAD model
- GLaDOS TTS model
- Vietnamese Piper TTS model (vi_VN-vais1000-medium)
- Kokoro TTS model
- Phonemizer model

## Configuration

Configuration files are located in the `configs/` directory. The main configuration file is `glados_config.yaml`.

### Configuration Options

```yaml
Glados:
  completion_url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
  model: "gemini-2.0-flash"
  api_key: "YOUR_API_KEY_HERE"
  api_type: "gemini"
  interruptible: false
  asr_model: "zipformer"
  wake_word: null
  voice: "vietnamese"
  announcement: "Your welcome message here"
  personality_preprompt:
    - system: |
        Your system prompt defining the assistant's personality and behavior
    - user: "Example user question"
    - assistant: "Example assistant response"
```

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key and paste it into your `glados_config.yaml` file

## Usage

### Start the Voice Assistant

```bash
uv run glados start
```

**With Custom Configuration:**

```bash
uv run glados start --config path/to/your/config.yaml
```

### Text-to-Speech Only

Convert text to speech without starting the full assistant:

```bash
uv run glados say "Hello, I am Soni!"
```

### Python API

You can also use the assistant programmatically:

```python
from glados.engine import Glados, GladosConfig

config = GladosConfig.from_yaml("configs/glados_config.yaml")

assistant = Glados.from_config(config)

assistant.start_listen_event_loop()
```

## Project Structure

```
soni-voice-assistant/
├── configs/                    
│   ├── assistant_config.yaml
│   └── glados_config.yaml
├── models/                    
│   ├── ASR/                 
│   └── TTS/                  
├── src/glados/                
│   ├── ASR/               
│   ├── TTS/                   
│   ├── utils/                 
│   ├── cli.py                 
│   └── engine.py             
├── scripts/                   
├── tests/                     
├── pyproject.toml            
└── README.md                 
```

## Troubleshooting

### Audio Issues

**No microphone detected:**
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**Permission denied on audio device:**
```bash
sudo usermod -a -G audio $USER
```

### Model Download Issues

If model downloads fail:
```bash
uv run glados download
```

### Import Errors

If you encounter import errors:
```bash
uv sync --reinstall
```

### Raspberry Pi Optimization

For optimal performance on Raspberry Pi 4B:

```bash
sudo apt install python3.12
export ONNXRUNTIME_PREFER_SYSTEM_LIB=1
```

## Acknowledgments

- Built on the GLaDOS voice assistant framework
- Vietnamese ASR powered by Sherpa-ONNX Zipformer models
- Vietnamese TTS powered by Piper voices project (vi_VN-vais1000-medium)
- AI responses powered by Google Gemini 2.0 Flash
- Voice Activity Detection by Silero VAD
