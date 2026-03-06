from __future__ import annotations

import os
from typing import Optional

import requests


def elevenlabs_tts(text: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL", model_id: str = "eleven_multilingual_v2") -> Optional[bytes]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print(f"[TTS disabled] {text}")
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content
