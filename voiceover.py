"""
Voiceover generation — same voice roster as the Video Ad Editor (:3002).

Voice values:
  "edge:<MicrosoftVoiceName>"  — free Edge TTS voices (edge-tts)
  "el:<elevenlabs_voice_id>"   — ElevenLabs voices from the account

generate_voiceover() returns {"path": mp3, "words": [{text, start, end}, ...]}.
Word timings drive the karaoke highlight in the renderer: Edge supplies them
via WordBoundary events, ElevenLabs via its /with-timestamps endpoint.

Both the MP3 and its word timings are cached in library/voiceovers keyed by
(voice, text) hash, so regenerating a video never re-bills the TTS API.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
VO_CACHE_DIR = BASE_DIR / "library" / "voiceovers"
EDITOR_ENV = Path("/Users/loris/Desktop/TOP Z/TopZ Dashboard/VideoAdEditor/server/.env")

def _find_edge_tts() -> str:
    import sys
    venv_bin = Path(sys.executable).parent / "edge-tts"
    if venv_bin.exists():
        return str(venv_bin)
    return shutil.which("edge-tts") or "/opt/homebrew/bin/edge-tts"


EDGE_TTS_BIN = _find_edge_tts()

# Same public fallback voice the editor uses when edge-tts dies (Adam,
# mirrors the edge default male voice).
ELEVEN_FALLBACK_VOICE = "pNInz6obpgDQGcFmaJgB"
ELEVEN_MODEL = "eleven_multilingual_v2"
ELEVEN_VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}

# Identical roster + labels to the editor UI (client/src/App.js).
STOCK_VOICES = [
    {"group": "US Male", "voices": [
        ("en-US-AndrewMultilingualNeural", "Andrew (Warm, Confident)"),
        ("en-US-BrianMultilingualNeural", "Brian (Casual, Sincere)"),
        ("en-US-ChristopherNeural", "Christopher (Authority)"),
        ("en-US-EricNeural", "Eric (Rational)"),
        ("en-US-GuyNeural", "Guy (Passionate)"),
        ("en-US-RogerNeural", "Roger (Lively)"),
        ("en-US-SteffanNeural", "Steffan (Rational)"),
    ]},
    {"group": "US Female", "voices": [
        ("en-US-AvaMultilingualNeural", "Ava (Caring, Friendly)"),
        ("en-US-EmmaMultilingualNeural", "Emma (Cheerful, Clear)"),
        ("en-US-JennyNeural", "Jenny (Friendly)"),
        ("en-US-AriaNeural", "Aria (Confident)"),
        ("en-US-MichelleNeural", "Michelle (Pleasant)"),
        ("en-US-AnaNeural", "Ana (Cute, Conversational)"),
    ]},
    {"group": "Canadian (US-Passing)", "voices": [
        ("en-CA-LiamNeural", "Liam (Canadian)"),
        ("en-CA-ClaraNeural", "Clara (Canadian)"),
    ]},
    {"group": "UK English", "voices": [
        ("en-GB-RyanNeural", "Ryan (UK)"),
        ("en-GB-SoniaNeural", "Sonia (UK)"),
        ("en-GB-ThomasNeural", "Thomas (UK)"),
        ("en-GB-LibbyNeural", "Libby (UK)"),
        ("en-GB-MaisieNeural", "Maisie (UK)"),
    ]},
    {"group": "Other", "voices": [
        ("en-AU-WilliamMultilingualNeural", "William (Australian)"),
        ("en-AU-NatashaNeural", "Natasha (Australian)"),
        ("en-NZ-MitchellNeural", "Mitchell (NZ)"),
        ("en-NZ-MollyNeural", "Molly (NZ)"),
        ("en-IE-ConnorNeural", "Connor (Irish)"),
        ("en-IE-EmilyNeural", "Emily (Irish)"),
    ]},
]


def _elevenlabs_key() -> str | None:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return key
    # Fallbacks: .env beside this module (VM deploys), then the editor's .env (local Mac)
    for env_file in (BASE_DIR / ".env", EDITOR_ENV):
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ELEVENLABS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


_eleven_cache: dict = {"ts": 0.0, "voices": []}


def _fetch_eleven_voices() -> list[dict]:
    """Live voice list from the ElevenLabs account, cached for 10 minutes."""
    if time.time() - _eleven_cache["ts"] < 600:
        return _eleven_cache["voices"]
    key = _elevenlabs_key()
    if not key:
        return []
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": key},
            timeout=15,
        )
        r.raise_for_status()
        voices = []
        for v in r.json().get("voices", []):
            labels = v.get("labels") or {}
            voices.append({
                "id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category"),
                "gender": labels.get("gender"),
            })
        # Own cloned/generated voices first, like the editor
        voices.sort(key=lambda v: 0 if v["category"] in ("cloned", "generated") else 1)
        _eleven_cache.update(ts=time.time(), voices=voices)
        return voices
    except Exception as e:
        print(f"ElevenLabs voice list unavailable: {e}")
        return _eleven_cache["voices"]


def get_voice_catalog() -> dict:
    return {
        "stock": [
            {"group": g["group"],
             "voices": [{"value": f"edge:{vid}", "label": label} for vid, label in g["voices"]]}
            for g in STOCK_VOICES
        ],
        "elevenlabs": _fetch_eleven_voices(),
    }


def _words_from_char_alignment(alignment: dict) -> list[dict]:
    """Group ElevenLabs per-character timings into per-word timings."""
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    words = []
    current, w_start, w_end = "", None, None
    for i, ch in enumerate(chars):
        if ch.isspace():
            if current:
                words.append({"text": current, "start": w_start, "end": w_end})
                current, w_start, w_end = "", None, None
            continue
        current += ch
        if w_start is None and i < len(starts):
            w_start = starts[i]
        if i < len(ends):
            w_end = ends[i]
    if current and w_start is not None:
        words.append({"text": current, "start": w_start, "end": w_end})
    return words


def _eleven_tts(text: str, voice_id: str, out_path: Path) -> list[dict]:
    """ElevenLabs TTS with character timings; returns per-word timings."""
    key = _elevenlabs_key()
    if not key:
        raise RuntimeError("ElevenLabs API key missing")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={"text": text, "model_id": ELEVEN_MODEL, "voice_settings": ELEVEN_VOICE_SETTINGS},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS failed ({r.status_code}): {r.text[:300]}")
    data = r.json()
    out_path.write_bytes(base64.b64decode(data["audio_base64"]))
    return _words_from_char_alignment(data.get("alignment") or {})


async def _edge_stream(text: str, voice_name: str, out_path: Path) -> list[dict]:
    """Stream Edge TTS audio + WordBoundary events (exact per-word timings)."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice_name, boundary="WordBoundary")
    audio = bytearray()
    words = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 1e7
            words.append({
                "text": chunk["text"],
                "start": start,
                "end": start + chunk["duration"] / 1e7,
            })
    if not audio:
        raise RuntimeError("No audio was received")
    out_path.write_bytes(bytes(audio))
    return words


def _edge_tts(text: str, voice_name: str, out_path: Path) -> list[dict]:
    """Edge TTS with the editor's retry logic; falls back to ElevenLabs Adam."""
    last_err = None
    for attempt in range(1, 7):
        try:
            return asyncio.run(_edge_stream(text, voice_name, out_path))
        except Exception as e:
            last_err = e
        time.sleep(attempt * 0.4)
    print(f"edge-tts failed after 6 attempts ({last_err}); falling back to ElevenLabs")
    return _eleven_tts(text, ELEVEN_FALLBACK_VOICE, out_path)


def strip_emoji(text: str) -> str:
    """
    Remove emoji (and their variation selectors) so the voice never reads them
    aloud — TTS engines otherwise pronounce them as "fire", "check mark", etc.
    The on-screen text keeps its emoji; only the spoken copy is stripped.
    """
    from renderer import EMOJI_RE

    cleaned = EMOJI_RE.sub(" ", text)
    # Collapse the gaps the emoji left behind, keeping paragraph breaks.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def generate_voiceover(text: str, voice: str) -> dict:
    """
    Return {"path": <mp3 path>, "words": [{text, start, end}, ...]} for the
    given text + voice, generating and caching it on first use.

    Emoji are stripped before the text reaches the TTS engine.
    """
    text = strip_emoji(text)
    VO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:20]
    out_path = VO_CACHE_DIR / f"vo_{key}.mp3"
    words_path = VO_CACHE_DIR / f"vo_{key}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        words = []
        try:
            words = json.loads(words_path.read_text())
        except (OSError, ValueError):
            pass
        return {"path": str(out_path), "words": words}

    tmp_path = out_path.with_suffix(".tmp.mp3")
    try:
        if voice.startswith("el:"):
            words = _eleven_tts(text, voice[3:], tmp_path)
        elif voice.startswith("edge:"):
            words = _edge_tts(text, voice[5:], tmp_path)
        else:
            raise ValueError(f"Unknown voice format: {voice}")
        tmp_path.rename(out_path)
        words_path.write_text(json.dumps(words))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {"path": str(out_path), "words": words}
