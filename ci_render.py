"""
CI render entrypoint (GitHub Actions).

Reads job parameters from env vars, pulls assets from R2, generates the
optional voiceover (R2-cached by (voice, text) hash so regenerates never
re-bill the TTS API), renders, and uploads the video back to R2.

Env:
  CELL_KEY, IMAGE_R2_KEY, VIDEO_TEXT              (required)
  SCROLL_SPEED, HOOK_FONT_SIZE, BODY_FONT_SIZE    (optional ints)
  VOICE            "" | "edge:<name>" | "el:<voice_id>"
  VOICE_VOLUME     0-100
  MUSIC_R2_KEY     "" | R2 key under music/
  MUSIC_VOLUME     0-100
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
  ELEVENLABS_API_KEY                              (for el: voices / fallback)
"""

import hashlib
import os
from pathlib import Path

import boto3

from renderer import render_video
from voiceover import generate_voiceover

R2_ENDPOINT = "https://346c911ecd013de0d51b44f77e5f2ec0.r2.cloudflarestorage.com"
R2_BUCKET = "scroll-generator"


def s3():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main():
    cell_key = os.environ["CELL_KEY"]
    image_r2_key = os.environ["IMAGE_R2_KEY"]
    text = os.environ["VIDEO_TEXT"]
    scroll_speed = int(os.environ.get("SCROLL_SPEED") or 80)
    hook_font_size = int(os.environ.get("HOOK_FONT_SIZE") or 52)
    body_font_size = int(os.environ.get("BODY_FONT_SIZE") or 38)
    voice = (os.environ.get("VOICE") or "").strip()
    voice_volume = max(0, min(100, int(os.environ.get("VOICE_VOLUME") or 100)))
    music_r2_key = (os.environ.get("MUSIC_R2_KEY") or "").strip()
    music_volume = max(0, min(100, int(os.environ.get("MUSIC_VOLUME") or 30)))

    client = s3()

    # Image
    ext = Path(image_r2_key).suffix or ".png"
    image_path = f"/tmp/input_image{ext}"
    client.download_file(R2_BUCKET, image_r2_key, image_path)
    print(f"Downloaded image {image_r2_key}")

    # Music (optional)
    music_path = None
    if music_r2_key:
        music_path = f"/tmp/music{Path(music_r2_key).suffix or '.mp3'}"
        client.download_file(R2_BUCKET, music_r2_key, music_path)
        print(f"Downloaded music {music_r2_key}")

    # Voiceover (optional, R2-cached)
    vo_path = None
    if voice:
        key = hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:20]
        vo_r2_key = f"voiceovers/vo_{key}.mp3"
        vo_path = f"/tmp/vo_{key}.mp3"
        try:
            client.download_file(R2_BUCKET, vo_r2_key, vo_path)
            print(f"Voiceover cache hit: {vo_r2_key}")
        except Exception:
            generated = generate_voiceover(text, voice)
            os.replace(generated, vo_path)
            client.upload_file(vo_path, R2_BUCKET, vo_r2_key,
                               ExtraArgs={"ContentType": "audio/mpeg"})
            print(f"Voiceover generated and cached: {vo_r2_key}")

    render_video(
        image_path=image_path,
        text=text,
        output_path="/tmp/output.mp4",
        scroll_speed=scroll_speed,
        hook_font_size=hook_font_size,
        body_font_size=body_font_size,
        music_path=music_path,
        music_volume=music_volume / 100.0,
        vo_path=vo_path,
        vo_volume=voice_volume / 100.0,
    )
    print("Video rendered successfully")

    client.upload_file("/tmp/output.mp4", R2_BUCKET,
                       f"videos/generated/{cell_key}.mp4",
                       ExtraArgs={"ContentType": "video/mp4"})
    print(f"Uploaded videos/generated/{cell_key}.mp4")


if __name__ == "__main__":
    main()
