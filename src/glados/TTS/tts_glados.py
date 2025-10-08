from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from pickle import load
from typing import Any
import numpy as np
from numpy.typing import NDArray
import onnxruntime 

try:
    import piper
    USING_PIPER_TTS = True
except ImportError:
    print("Warning: piper-tts not available, using vn_phonemizer as fallback")
    USING_PIPER_TTS = False
    from .vn_phonemizer import Phonemizer as VnPhonemizer

from ..utils.resources import resource_path
from .phonemizer import Phonemizer


class Synthesizer:
    """Synthesizer, based on the VITS model.

    Trained using the Piper project (https://github.com/rhasspy/piper)

    Attributes:
    -----------
    session: onnxruntime.InferenceSession
        The loaded VITS model.
    phonemizer: Phonemizer
        Custom phonemizer for converting text to phonemes.

    Methods:
    --------
    __init__(self, model_path, config_path):
        Initializes the Synthesizer class, loading the VITS model.

    audio_float_to_int16(self, audio, max_wav_value):
        Converts audio from float to int16 format.

    phonemize(self, text):
        Converts text to phonemes using the custom Phonemizer.

    phonemes_to_ids(self, phonemes):
        Converts the given phonemes to ids.

    synthesize_audio(self, text):
        Generates speech audio from the given text.
    """

    # Constants
    MAX_WAV_VALUE = 32767.0

    # Settings
    MODEL_PATH = resource_path("models/TTS/vi_VN-vais1000-medium.onnx")
    CONFIG_PATH = resource_path("models/TTS/vi_VN-vais1000-medium.onnx.json")

    # Conversions
    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def __init__(
        self, 
        model_path: Path = MODEL_PATH, 
        config_path: Path = CONFIG_PATH,
        phonemizer: Phonemizer | None = None
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.session = onnxruntime.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        self.sample_rate = self.config["audio"]["sample_rate"]
        
        # Initialize phonemizer based on availability
        if not USING_PIPER_TTS:
            # Use Vietnamese phonemizer as fallback
            if phonemizer is None:
                self.phonemizer = VnPhonemizer()
            else:
                self.phonemizer = phonemizer
        else:
            if phonemizer is not None:
                self.phonemizer = phonemizer

    def audio_float_to_int16(
        self, 
        audio: np.ndarray, 
        max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def phonemize(self, text: str) -> list[str] | list[list[str]]:
        """
        Convert text to phonemes.
        
        Args:
            text: Input text to phonemize
            
        Returns:
            List of phoneme strings or list of phoneme lists
        """
        if USING_PIPER_TTS:
            # Use piper-tts
            try:
                # Create a Piper voice instance for phonemization
                voice = piper.PiperVoice.load(
                    model_path=str(self.MODEL_PATH), 
                    config_path=str(self.CONFIG_PATH)
                )
                
                # Get phonemes from the voice
                phonemes = voice.phonemize(text)
                return phonemes
                    
            except Exception as e:
                print(f"Warning: piper-tts phonemization failed: {e}")
                # Fall back to Vietnamese phonemizer
                if hasattr(self, 'phonemizer'):
                    texts = [text] if isinstance(text, str) else text
                    phoneme_strings = self.phonemizer.convert_to_phonemes(texts, lang="en_us")
                    return phoneme_strings
                else:
                    return [text.split()]  # Basic fallback
        else:
            # Use Vietnamese phonemizer as fallback
            texts = [text] if isinstance(text, str) else text
            phoneme_strings = self.phonemizer.convert_to_phonemes(texts, lang="en_us")
            return phoneme_strings

    def phonemes_to_ids(self, phonemes: str | list[str]) -> list[int]:
        """
        Convert phoneme string or list to phoneme IDs.
        
        Args:
            phonemes: String of phonemes or list of phonemes
            
        Returns:
            List of phoneme IDs
        """
        id_map = self.config["phoneme_id_map"]
        ids = [id_map["^"][0]]  # Start of sentence (BOS)
        
        # Handle both string and list input
        phoneme_list = list(phonemes) if isinstance(phonemes, (str, list)) else phonemes
        
        for phoneme in phoneme_list:
            if phoneme in id_map:
                ids.extend(id_map[phoneme])
            ids.extend(id_map["_"])  # PAD between phonemes
            
        ids.append(id_map["$"][0])  # End of sentence (EOS)
        return ids

    def synthesize_audio(self, text: str) -> np.ndarray:
        """
        Synthesize audio from text.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Audio array as int16
        """
        # Get phonemes for the text
        phoneme_strings = self.phonemize(text)

        audio_segments = []
        for phoneme_str in phoneme_strings:
            # Convert phonemes to IDs
            phoneme_ids = self.phonemes_to_ids(phoneme_str)
            
            # Prepare inputs for ONNX model
            text_tensor = np.array([phoneme_ids], dtype=np.int64)
            text_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)  # default scales

            inputs = {
                "input": text_tensor,
                "input_lengths": text_lengths,
                "scales": scales,
            }

            # Add speaker ID if multi-speaker model
            if self.config["num_speakers"] > 1:
                inputs["sid"] = np.array([0], dtype=np.int64)  # default speaker ID

            # Run inference
            audio = self.session.run(None, inputs)[0].squeeze()
            print("AUDIO: ", audio)
            
            # Convert to int16
            audio_int16 = self.audio_float_to_int16(audio)
            audio_segments.append(audio_int16)

        # Concatenate all audio segments
        full_audio = np.concatenate(audio_segments)
        return full_audio

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session"):
            del self.session