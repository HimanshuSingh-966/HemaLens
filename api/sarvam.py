"""
HemaLens — Sarvam AI API Wrapper
=================================
Handles two Sarvam AI APIs:
  1. Translate API  — English → regional Indian language
  2. TTS API        — text → audio bytes (WAV)

Sarvam docs: https://docs.sarvam.ai
Get API key : https://dashboard.sarvam.ai
"""
import os
import requests
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to OS env vars

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TTS_URL       = "https://api.sarvam.ai/text-to-speech"

# Sarvam language codes
LANGUAGE_CODES = {
    "hindi":     "hi-IN",
    "tamil":     "ta-IN",
    "telugu":    "te-IN",
    "kannada":   "kn-IN",
    "malayalam": "ml-IN",
    "bengali":   "bn-IN",
    "marathi":   "mr-IN",
    "gujarati":  "gu-IN",
    "punjabi":   "pa-IN",
    "english":   "en-IN",
}

# Sarvam TTS speaker voices
# patient mode → warm, clear, calm voice
# clinical mode → neutral, professional voice
SPEAKERS = {
    "patient":  "meera",   # warm female voice
    "clinical": "amol",    # neutral male voice
}


def translate_text(text: str, target_language: str) -> str:
    """
    Translate English text to a regional Indian language using Sarvam AI.

    Args:
        text:            English input text
        target_language: one of LANGUAGE_CODES keys (e.g. 'hindi', 'tamil')

    Returns:
        Translated text string. Falls back to original English on error.
    """
    if target_language == "english" or not target_language:
        return text

    if not SARVAM_API_KEY:
        print("⚠️  SARVAM_API_KEY not set — returning English text")
        return text

    lang_code = LANGUAGE_CODES.get(target_language.lower())
    if not lang_code:
        return text

    try:
        response = requests.post(
            SARVAM_TRANSLATE_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input":           text,
                "source_language": "en-IN",
                "target_language": lang_code,
                "speaker_gender":  "Female",
                "mode":            "formal",
                "enable_preprocessing": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("translated_text", text)
    except Exception as e:
        print(f"⚠️  Sarvam translate error: {e} — falling back to English")
        return text


def text_to_speech(text: str, language: str, mode: str = "patient") -> Optional[bytes]:
    """
    Convert text to speech audio using Sarvam AI TTS.

    Args:
        text:     Text to speak (in the target language)
        language: one of LANGUAGE_CODES keys
        mode:     'patient' or 'clinical' — controls speaker voice

    Returns:
        WAV audio bytes, or None if TTS fails.
    """
    if not SARVAM_API_KEY:
        print("⚠️  SARVAM_API_KEY not set — TTS unavailable")
        return None

    lang_code = LANGUAGE_CODES.get(language.lower(), "hi-IN")
    speaker   = SPEAKERS.get(mode, "meera")

    # Sarvam TTS has a ~500 char limit per request — split if needed
    chunks = _split_text(text, max_chars=490)
    audio_parts = []

    for chunk in chunks:
        try:
            response = requests.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs":          [chunk],
                    "target_language": lang_code,
                    "speaker":         speaker,
                    "pitch":           0,
                    "pace":            1.0,
                    "loudness":        1.5,
                    "speech_sample_rate": 8000,
                    "enable_preprocessing": True,
                    "model":           "bulbul:v1",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            # Sarvam returns base64-encoded audio
            import base64
            audio_b64 = data.get("audios", [""])[0]
            if audio_b64:
                audio_parts.append(base64.b64decode(audio_b64))
        except Exception as e:
            print(f"⚠️  Sarvam TTS error on chunk: {e}")
            continue

    if not audio_parts:
        return None

    # Concatenate WAV parts — strip headers from all but the first
    return _concat_wav(audio_parts)


def _split_text(text: str, max_chars: int = 490) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text[:max_chars]]


def _concat_wav(parts: list[bytes]) -> bytes:
    """Concatenate multiple WAV byte strings into one."""
    if len(parts) == 1:
        return parts[0]
    # Use first part's header, append raw PCM from subsequent parts
    # WAV header is 44 bytes
    combined_pcm = b"".join(p[44:] for p in parts)
    header = bytearray(parts[0][:44])
    # Update data chunk size (bytes 40-43)
    data_size = len(combined_pcm)
    header[40:44] = data_size.to_bytes(4, "little")
    # Update RIFF chunk size (bytes 4-7)
    header[4:8] = (36 + data_size).to_bytes(4, "little")
    return bytes(header) + combined_pcm