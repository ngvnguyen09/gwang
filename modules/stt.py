import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("stt_module")

# The user mentioned using base or small model, beam_size=5, and language="vi"
MODEL_SIZE = "small"
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribes raw PCM 16kHz 16-bit mono audio to text using Faster-Whisper.
    """
    if not audio_data:
        return ""
        
    try:
        # Faster-whisper input expects numpy array of float32 ideally normalized
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        logger.info("Starting Whisper transcription...")
        segments, info = whisper_model.transcribe(
            audio_np, 
            beam_size=5, 
            language="vi"
        )
        
        text = "".join([segment.text for segment in segments])
        logger.info(f"Transcription result: {text.strip()}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return ""
