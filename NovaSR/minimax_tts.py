"""
MiniMax TTS integration with NovaSR audio super-resolution.

Generates speech via the MiniMax TTS API and enhances it to 48 kHz
using NovaSR's FastSR model — leveraging NovaSR's ability to enhance
TTS output quality with nearly zero computational overhead.

Environment variable:
    MINIMAX_API_KEY: Your MiniMax API key  (https://api.minimax.io)

Example::

    from NovaSR import FastSR
    from NovaSR.minimax_tts import generate_and_enhance
    import torchaudio

    upsampler = FastSR()
    enhanced = generate_and_enhance("Hello, world!", upsampler=upsampler)
    torchaudio.save("enhanced.wav", enhanced.cpu(), 48000)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from NovaSR import FastSR

MINIMAX_TTS_BASE_URL = "https://api.minimax.io"
DEFAULT_TTS_MODEL = "speech-2.8-hd"
DEFAULT_VOICE = "English_Graceful_Lady"

MINIMAX_TTS_VOICES = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
]


def generate_speech(
    text: str,
    *,
    api_key: Optional[str] = None,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    base_url: str = MINIMAX_TTS_BASE_URL,
) -> bytes:
    """Generate speech audio using the MiniMax TTS API.

    Args:
        text: Text to synthesize.
        api_key: MiniMax API key.  Falls back to the ``MINIMAX_API_KEY``
            environment variable.
        voice: Voice ID to use.  See :data:`MINIMAX_TTS_VOICES` for the
            built-in options.
        model: TTS model name.  Use ``'speech-2.8-hd'`` (default) for
            highest quality or ``'speech-2.8-turbo'`` for faster synthesis.
        base_url: MiniMax API base URL.  Override for private deployments.

    Returns:
        Raw MP3 audio bytes returned by the API.

    Raises:
        ValueError: If no API key is provided or discoverable.
        ImportError: If the ``requests`` package is not installed.
        requests.HTTPError: If the API request fails.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for MiniMax TTS integration. "
            "Install it with: pip install requests"
        ) from exc

    resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A MiniMax API key is required. "
            "Set the MINIMAX_API_KEY environment variable or pass api_key= explicitly."
        )

    response = requests.post(
        f"{base_url}/v1/t2a_v2",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_key}",
        },
        json={
            "model": model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": voice,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        },
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    # Parse SSE stream and collect hex-encoded audio chunks.
    audio_chunks: list[bytes] = []
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data:"):
            continue
        json_str = decoded[5:].strip()
        if not json_str or json_str == "[DONE]":
            continue
        try:
            event_data = json.loads(json_str)
            audio_hex: Optional[str] = event_data.get("data", {}).get("audio")
            if audio_hex:
                audio_chunks.append(bytes.fromhex(audio_hex))
        except (json.JSONDecodeError, ValueError):
            continue

    return b"".join(audio_chunks)


def generate_and_enhance(
    text: str,
    upsampler: Optional["FastSR"] = None,
    *,
    api_key: Optional[str] = None,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    base_url: str = MINIMAX_TTS_BASE_URL,
) -> torch.Tensor:
    """Generate speech with MiniMax TTS and enhance it to 48 kHz with NovaSR.

    This combines MiniMax's expressive TTS with NovaSR's ultra-fast
    audio super-resolution, producing crisp 48 kHz speech output at a
    fraction of the cost of larger models.

    Args:
        text: Text to synthesize.
        upsampler: A :class:`~NovaSR.FastSR` instance.  One is created
            automatically (downloading the model from Hugging Face) when
            ``None`` is passed.
        api_key: MiniMax API key.  Falls back to the ``MINIMAX_API_KEY``
            environment variable.
        voice: Voice ID to use.  See :data:`MINIMAX_TTS_VOICES` for the
            built-in options.
        model: TTS model name.  Use ``'speech-2.8-hd'`` (default) or
            ``'speech-2.8-turbo'``.
        base_url: MiniMax API base URL.

    Returns:
        Enhanced audio tensor at 48 kHz with shape ``[1, num_samples]``.

    Example::

        import torchaudio
        from NovaSR import FastSR
        from NovaSR.minimax_tts import generate_and_enhance

        sr = FastSR()
        audio = generate_and_enhance("Hello, how are you?", upsampler=sr)
        torchaudio.save("output.wav", audio.cpu(), 48000)
    """
    from NovaSR import FastSR

    if upsampler is None:
        upsampler = FastSR()

    audio_bytes = generate_speech(
        text, api_key=api_key, voice=voice, model=model, base_url=base_url
    )

    # Write to a temporary file so FastSR.load_audio can process it.
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        lowres_wav = upsampler.load_audio(tmp_path)
        return upsampler.infer(lowres_wav)
    finally:
        os.unlink(tmp_path)
