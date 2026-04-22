"""Unit tests for NovaSR.minimax_tts module.

All tests mock external dependencies (the MiniMax API and FastSR model)
so they run offline without any GPU or API credentials.
"""
from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sse_lines(*audio_hexes: str) -> list[bytes]:
    """Return SSE byte lines for a sequence of audio hex chunks."""
    lines: list[bytes] = []
    for i, hex_chunk in enumerate(audio_hexes):
        status = 2 if i == len(audio_hexes) - 1 else 1
        data = json.dumps({"data": {"audio": hex_chunk, "status": status}})
        lines.append(f"data: {data}".encode())
    lines.append(b"data: [DONE]")
    return lines


def _mock_wav_bytes(sample_rate: int = 16000, num_samples: int = 1600) -> bytes:
    """Return a valid WAV file as bytes (1-channel, silent)."""
    buf = io.BytesIO()
    dummy = torch.zeros(1, num_samples)
    torchaudio.save(buf, dummy, sample_rate, format="wav")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# generate_speech
# ---------------------------------------------------------------------------

class TestGenerateSpeech:
    @patch("requests.post")
    def test_returns_concatenated_audio_bytes(self, mock_post: MagicMock) -> None:
        """Concatenates hex-decoded chunks from the SSE stream."""
        from NovaSR.minimax_tts import generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("dead", "beef")

        result = generate_speech("Hello", api_key="test_key")

        assert result == bytes.fromhex("deadbeef")

    @patch("requests.post")
    def test_sends_correct_api_url(self, mock_post: MagicMock) -> None:
        """Posts to the t2a_v2 endpoint."""
        from NovaSR.minimax_tts import MINIMAX_TTS_BASE_URL, generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("aa")

        generate_speech("Hello", api_key="key")

        url = mock_post.call_args.args[0]
        assert url == f"{MINIMAX_TTS_BASE_URL}/v1/t2a_v2"

    @patch("requests.post")
    def test_authorization_header_uses_api_key(self, mock_post: MagicMock) -> None:
        """Passes the API key in the Authorization header."""
        from NovaSR.minimax_tts import generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("00")

        generate_speech("Hello", api_key="my_secret_key")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my_secret_key"

    @patch("requests.post")
    def test_falls_back_to_env_api_key(self, mock_post: MagicMock) -> None:
        """Uses MINIMAX_API_KEY environment variable when api_key is not given."""
        from NovaSR.minimax_tts import generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("ff")

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env_key"}):
            generate_speech("Hello")

        headers = mock_post.call_args.kwargs["headers"]
        assert "env_key" in headers["Authorization"]

    def test_raises_when_no_api_key(self) -> None:
        """Raises ValueError when no API key is available."""
        from NovaSR.minimax_tts import generate_speech

        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                generate_speech("Hello")

    @patch("requests.post")
    def test_model_and_voice_forwarded(self, mock_post: MagicMock) -> None:
        """Passes model and voice_id to the request payload."""
        from NovaSR.minimax_tts import generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("ab")

        generate_speech(
            "Hi",
            api_key="k",
            voice="English_Persuasive_Man",
            model="speech-2.8-turbo",
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "speech-2.8-turbo"
        assert payload["voice_setting"]["voice_id"] == "English_Persuasive_Man"

    @patch("requests.post")
    def test_custom_base_url(self, mock_post: MagicMock) -> None:
        """Respects a custom base_url."""
        from NovaSR.minimax_tts import generate_speech

        mock_post.return_value.iter_lines.return_value = _make_sse_lines("cc")

        generate_speech("Hi", api_key="k", base_url="https://custom.api.io")

        url = mock_post.call_args.args[0]
        assert url.startswith("https://custom.api.io")

    @patch("requests.post")
    def test_skips_non_data_lines(self, mock_post: MagicMock) -> None:
        """Ignores SSE comment and event lines that don't start with 'data:'."""
        from NovaSR.minimax_tts import generate_speech

        extra_lines = [
            b": keep-alive",
            b"event: message",
            b"",
            *_make_sse_lines("cafe"),
        ]
        mock_post.return_value.iter_lines.return_value = extra_lines

        result = generate_speech("Hello", api_key="k")
        assert result == bytes.fromhex("cafe")

    @patch("requests.post")
    def test_raises_import_error_without_requests(self, _: MagicMock) -> None:
        """Raises ImportError with install hint when requests is missing."""
        import builtins

        real_import = builtins.__import__

        def block_requests(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        from NovaSR.minimax_tts import generate_speech

        with patch("builtins.__import__", side_effect=block_requests):
            with pytest.raises(ImportError, match="pip install requests"):
                generate_speech("Hello", api_key="k")


# ---------------------------------------------------------------------------
# generate_and_enhance
# ---------------------------------------------------------------------------

class TestGenerateAndEnhance:
    @patch("NovaSR.minimax_tts.generate_speech")
    def test_calls_generate_speech_then_upsampler(
        self, mock_generate: MagicMock
    ) -> None:
        """Invokes generate_speech, loads the audio, and runs inference."""
        from NovaSR.minimax_tts import generate_and_enhance

        mock_generate.return_value = _mock_wav_bytes()

        mock_upsampler = MagicMock()
        mock_upsampler.infer.return_value = torch.zeros(1, 48000)

        result = generate_and_enhance("Hello", upsampler=mock_upsampler, api_key="key")

        mock_generate.assert_called_once()
        mock_upsampler.load_audio.assert_called_once()
        mock_upsampler.infer.assert_called_once()
        assert isinstance(result, torch.Tensor)

    @patch("NovaSR.minimax_tts.generate_speech")
    def test_temp_file_cleaned_up(self, mock_generate: MagicMock) -> None:
        """Removes the temporary audio file after inference."""
        from NovaSR.minimax_tts import generate_and_enhance

        mock_generate.return_value = _mock_wav_bytes()

        captured_path: list[str] = []

        mock_upsampler = MagicMock()
        mock_upsampler.infer.return_value = torch.zeros(1, 48000)

        def capture_load(path: str) -> MagicMock:
            captured_path.append(path)
            return MagicMock()

        mock_upsampler.load_audio.side_effect = capture_load

        generate_and_enhance("Hello", upsampler=mock_upsampler, api_key="k")

        assert captured_path, "load_audio was not called"
        assert not os.path.exists(captured_path[0]), "Temp file was not removed"

    @patch("NovaSR.minimax_tts.generate_speech")
    def test_temp_file_cleaned_up_on_inference_error(
        self, mock_generate: MagicMock
    ) -> None:
        """Removes the temp file even when inference raises an exception."""
        from NovaSR.minimax_tts import generate_and_enhance

        mock_generate.return_value = _mock_wav_bytes()

        captured_path: list[str] = []

        mock_upsampler = MagicMock()

        def capture_load(path: str) -> MagicMock:
            captured_path.append(path)
            return MagicMock()

        mock_upsampler.load_audio.side_effect = capture_load
        mock_upsampler.infer.side_effect = RuntimeError("GPU OOM")

        with pytest.raises(RuntimeError, match="GPU OOM"):
            generate_and_enhance("Hello", upsampler=mock_upsampler, api_key="k")

        assert captured_path
        assert not os.path.exists(captured_path[0]), "Temp file was not removed"

    @patch("NovaSR.minimax_tts.generate_speech")
    def test_passes_kwargs_to_generate_speech(
        self, mock_generate: MagicMock
    ) -> None:
        """Forwards voice, model, base_url, and api_key to generate_speech."""
        from NovaSR.minimax_tts import generate_and_enhance

        mock_generate.return_value = _mock_wav_bytes()
        mock_upsampler = MagicMock()
        mock_upsampler.infer.return_value = torch.zeros(1, 48000)

        generate_and_enhance(
            "Hello",
            upsampler=mock_upsampler,
            api_key="my_key",
            voice="English_Lucky_Robot",
            model="speech-2.8-turbo",
            base_url="https://custom.io",
        )

        mock_generate.assert_called_once_with(
            "Hello",
            api_key="my_key",
            voice="English_Lucky_Robot",
            model="speech-2.8-turbo",
            base_url="https://custom.io",
        )
