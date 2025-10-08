from pathlib import Path
from typing import List, Optional, Union
import urllib.request
import os
import zipfile

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


class ZipformerAudioTranscriber:
    """
    Audio transcriber using Sherpa ONNX Zipformer model for Vietnamese speech recognition.
    
    This implementation uses the Vietnamese Zipformer model from:
    https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20
    """
    
    MODEL_BASE_PATH = resource_path("models/ASR/zipformer")
    ENCODER_PATH = MODEL_BASE_PATH / "encoder-epoch-12-avg-8.onnx"
    DECODER_PATH = MODEL_BASE_PATH / "decoder-epoch-12-avg-8.onnx"
    JOINER_PATH = MODEL_BASE_PATH / "joiner-epoch-12-avg-8.onnx"
    TOKENS_PATH = MODEL_BASE_PATH / "tokens.txt"
    BPE_MODEL_PATH = MODEL_BASE_PATH / "bpe.model"
    
    # Model download URLs
    MODEL_URLS = {
        "encoder": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/main/encoder-epoch-12-avg-8.onnx",
        "decoder": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/main/decoder-epoch-12-avg-8.onnx",
        "joiner": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/main/joiner-epoch-12-avg-8.onnx",
        "tokens": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/main/tokens.txt",
        "bpe": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/main/bpe.model"
    }
    
    SAMPLE_RATE = 16000

    def __init__(
        self,
        encoder_path: Optional[Path] = None,
        decoder_path: Optional[Path] = None,
        joiner_path: Optional[Path] = None,
        tokens_path: Optional[Path] = None,
        auto_download: bool = True,
    ) -> None:
        """
        Initialize a ZipformerAudioTranscriber with Sherpa ONNX Zipformer model.

        Parameters:
            encoder_path (Path, optional): Path to the encoder ONNX model file.
            decoder_path (Path, optional): Path to the decoder ONNX model file.
            joiner_path (Path, optional): Path to the joiner ONNX model file.
            tokens_path (Path, optional): Path to the tokens file.
            auto_download (bool): Whether to automatically download models if not found.

        The Zipformer architecture uses three separate models:
        - Encoder: Processes audio features
        - Decoder: Generates text predictions
        - Joiner: Combines encoder and decoder outputs
        """
        
        # Use default paths if not provided
        self.encoder_path = encoder_path or self.ENCODER_PATH
        self.decoder_path = decoder_path or self.DECODER_PATH
        self.joiner_path = joiner_path or self.JOINER_PATH
        self.tokens_path = tokens_path or self.TOKENS_PATH
        
        # Download models if they don't exist and auto_download is True
        if auto_download:
            self._ensure_models_exist()
        
        # Initialize ONNX Runtime providers
        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        # Create ONNX sessions for each model component
        sess_options = ort.SessionOptions()
        
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        self.joiner_session = ort.InferenceSession(
            str(self.joiner_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        # Load vocabulary
        self.vocab = self._load_vocabulary(self.tokens_path)
        self.blank_id = 0  # Typically blank token is at index 0
        
    def _ensure_models_exist(self) -> None:
        """Download model files if they don't exist."""
        # Create model directory if it doesn't exist
        self.MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
        
        model_files = {
            "encoder": self.encoder_path,
            "decoder": self.decoder_path,
            "joiner": self.joiner_path,
            "tokens": self.tokens_path,
            "bpe": self.BPE_MODEL_PATH,
        }
        
        for model_name, file_path in model_files.items():
            if not file_path.exists():
                print(f"Downloading {model_name} model...")
                url = self.MODEL_URLS[model_name]
                self._download_file(url, file_path)
                print(f"Downloaded {model_name} to {file_path}")
    
    def _download_file(self, url: str, destination: Path) -> None:
        """Download a file from URL to destination."""
        try:
            urllib.request.urlretrieve(url, destination)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    def _load_vocabulary(self, tokens_file: Path) -> dict[int, str]:
        """
        Load token vocabulary from a file.

        Parameters:
            tokens_file (Path): Path to the file containing tokens.

        Returns:
            dict[int, str]: A dictionary mapping token indices to token strings.
        """
        vocab = {}
        with open(tokens_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by the last space to handle tokens with spaces
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    token, idx_str = parts
                    try:
                        idx = int(idx_str)
                        vocab[idx] = token
                    except ValueError:
                        continue
        return vocab

    def _simple_resample(self, audio: NDArray[np.float32], orig_sr: int, target_sr: int) -> NDArray[np.float32]:
        """Simple linear interpolation resampling."""
        if orig_sr == target_sr:
            return audio
            
        # Calculate new length
        new_length = int(len(audio) * target_sr / orig_sr)
        
        # Linear interpolation
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio)
        
        return resampled.astype(np.float32)

    def _hann_window(self, n: int) -> NDArray[np.float32]:
        """Create Hann window."""
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

    def _mel_filter_bank(self, n_fft: int, n_mels: int, sample_rate: int = 16000, fmin: float = 0.0, fmax: float = None) -> NDArray[np.float32]:
        """Create mel filter bank."""
        if fmax is None:
            fmax = sample_rate / 2.0
            
        # Convert to mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin numbers
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        # Create filter bank
        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            for j in range(left, center):
                if center != left:
                    fbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    fbank[i - 1, j] = (right - j) / (right - center)
                    
        return fbank.astype(np.float32)

    def _compute_mel_spectrogram(self, audio: NDArray[np.float32], n_fft: int = 512, hop_length: int = 160, n_mels: int = 80) -> NDArray[np.float32]:
        """Compute mel-spectrogram from audio."""
        # Add padding
        pad_length = n_fft // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')
        
        # Frame the audio
        n_frames = (len(audio_padded) - n_fft) // hop_length + 1
        frames = np.zeros((n_frames, n_fft))
        
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = audio_padded[start:start + n_fft]
        
        # Apply window
        window = self._hann_window(n_fft)
        frames = frames * window
        
        # Compute STFT using numpy FFT
        stft = np.fft.rfft(frames, n=n_fft, axis=1)
        magnitude = np.abs(stft)
        power_spec = magnitude ** 2
        
        # Apply mel filter bank
        mel_basis = self._mel_filter_bank(n_fft, n_mels, self.SAMPLE_RATE)
        mel_spec = np.dot(power_spec, mel_basis.T)
        
        # Convert to log scale
        log_mel_spec = np.log(np.maximum(mel_spec, 1e-10))
        
        return log_mel_spec.astype(np.float32)

    def _preprocess_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Preprocess audio for the Zipformer model.
        
        Parameters:
            audio (NDArray[np.float32]): Input audio signal.
            
        Returns:
            NDArray[np.float32]: Preprocessed audio features (log-mel filterbank).
        """
        # Ensure audio is float32 and normalize
        audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95
            
        # Compute mel-spectrogram features
        features = self._compute_mel_spectrogram(audio, n_fft=512, hop_length=160, n_mels=80)
        
        return features

    def _run_encoder(self, audio: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Run the encoder model on audio features.
        
        Parameters:
            audio (NDArray[np.float32]): Preprocessed audio signal.
            
        Returns:
            tuple: (encoder_output, encoder_output_length)
        """
        # Prepare inputs for encoder
        audio_length = np.array([audio.shape[0]], dtype=np.int64)
        audio = np.expand_dims(audio, axis=0)  # Add batch dimension
        
        encoder_inputs = {
            "x": audio,
            "x_lens": audio_length
        }
        
        # Run encoder
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        encoder_out = encoder_outputs[0]  # [batch, time, feature_dim]
        encoder_out_lens = encoder_outputs[1]  # [batch]
        
        return encoder_out, encoder_out_lens

    def _greedy_search(self, encoder_out: NDArray[np.float32], encoder_out_lens: NDArray[np.int64]) -> List[int]:
        """
        Perform greedy search decoding.
        
        Parameters:
            encoder_out (NDArray[np.float32]): Encoder output features.
            encoder_out_lens (NDArray[np.int64]): Encoder output lengths.
            
        Returns:
            List[int]: Decoded token sequence.
        """
        batch_size, max_len, encoder_dim = encoder_out.shape
        device_id = 0  # Assuming single device
        
        # Initialize decoder state with context
        # The decoder expects a 2-token context: [blank, blank] initially
        decoder_input = np.array([[self.blank_id, self.blank_id]], dtype=np.int64)  # [batch, 2]
        decoder_outputs = self.decoder_session.run(None, {"y": decoder_input})
        decoder_out = decoder_outputs[0]  # [batch, 1, decoder_dim]

        hyp = []
        
        for t in range(encoder_out_lens[0]):
            # Get current encoder output
            current_encoder_out = encoder_out[0, t, :]  # [encoder_dim] - remove batch and time dims
            current_encoder_out = np.expand_dims(current_encoder_out, axis=0)  # [1, encoder_dim]
            
            # Get current decoder output - check dimensions first
            if decoder_out.ndim == 3:
                current_decoder_out = decoder_out[0, 0, :]  # [decoder_dim] - 3D case
            else:
                current_decoder_out = decoder_out[0, :]  # [decoder_dim] - 2D case
            current_decoder_out = np.expand_dims(current_decoder_out, axis=0)  # [1, decoder_dim]
            
            # Prepare joiner inputs
            joiner_inputs = {
                "encoder_out": current_encoder_out,
                "decoder_out": current_decoder_out
            }
            
            # Run joiner
            joiner_outputs = self.joiner_session.run(None, joiner_inputs)
            logits = joiner_outputs[0]  # Output shape varies
            
            # Get the most probable token - handle different output shapes
            if logits.ndim == 3:
                y = np.argmax(logits[0, 0, :])  # [batch, time, vocab] -> [vocab]
            elif logits.ndim == 2:
                y = np.argmax(logits[0, :])     # [batch, vocab] -> [vocab]
            else:
                y = np.argmax(logits)           # [vocab]
            
            if y != self.blank_id:
                hyp.append(y)
                # Update decoder state with new token and previous token
                prev_token = decoder_input[0, -1]  # Get the last token
                decoder_input = np.array([[prev_token, y]], dtype=np.int64)  # [batch, 2]
                decoder_outputs = self.decoder_session.run(None, {"y": decoder_input})
                decoder_out = decoder_outputs[0]
        
        return hyp  # Return the decoded tokens

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text.
        
        Parameters:
            token_ids (List[int]): List of token IDs.
            
        Returns:
            str: Decoded text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                token = self.vocab[token_id]
                if token not in ["<blk>", "<unk>", "<s>", "</s>"]:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = "".join(tokens)
        
        # Handle subword tokens (if any)
        text = text.replace("▁", " ").strip()
        
        return text

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribe audio to text using the Zipformer model.

        Parameters:
            audio (NDArray[np.float32]): Input audio signal.

        Returns:
            str: Transcribed text.
        """
        # Preprocess audio
        processed_audio = self._preprocess_audio(audio)
        
        # Run encoder
        encoder_out, encoder_out_lens = self._run_encoder(processed_audio)
        
        # Perform greedy search decoding
        token_ids = self._greedy_search(encoder_out, encoder_out_lens)
        
        # Decode tokens to text
        transcription = self._decode_tokens(token_ids)
        
        return transcription

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Parameters:
            audio_path (str): Path to the audio file.

        Returns:
            str: Transcribed text.
        """
        # Load audio file
        audio, sr = sf.read(audio_path, dtype="float32")
        
        # Resample to 16kHz if needed
        if sr != self.SAMPLE_RATE:
            audio = self._simple_resample(audio, sr, self.SAMPLE_RATE)
        
        return self.transcribe(audio)

    def __del__(self) -> None:
        """Clean up ONNX sessions to prevent context leaks."""
        for session_attr in ["encoder_session", "decoder_session", "joiner_session"]:
            if hasattr(self, session_attr):
                delattr(self, session_attr)