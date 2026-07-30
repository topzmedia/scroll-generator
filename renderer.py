"""
Video Renderer — creates scrolling text videos from image + text.

Pipeline:
1. Pillow composites a tall image: photo on top → text below on black
2. ffmpeg pans down over it to produce a 1080×1920 MP4
"""

import os
import subprocess
import textwrap
from pathlib import Path

def probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr}")
    return float(result.stdout.strip())

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 60
IMAGE_HEIGHT_RATIO = 0.35  # image takes top 35% of viewport
HOOK_FONT_SIZE = 52
BODY_FONT_SIZE = 38
LINE_SPACING_FACTOR = 1.5
SCROLL_SPEED = int(38 * 1.5 * 0.8)  # ~0.8 lines/sec (46 px/s, 20% slower)
PADDING_X = 60
TEXT_GAP = 60  # gap between image bottom and first text line
PARAGRAPH_GAP = 40  # extra gap between paragraphs
HOLD_DURATION = 4  # seconds to freeze on last frame
START_DELAY = 1.2  # seconds to wait before scrolling begins
FOOTER_MARGIN = 0.20  # last text stops 20% from viewport bottom

FONT_DIR = Path(__file__).parent / "fonts"
FONT_REGULAR = FONT_DIR / "Poppins-Regular.ttf"
FONT_BOLD = FONT_DIR / "Poppins-SemiBold.ttf"


def _load_font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    """Load a bundled font, falling back to default if missing."""
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        # Fallback: try system DejaVu (available in most Docker images)
        for fallback in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            if os.path.exists(fallback):
                return ImageFont.truetype(fallback, size)
        return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    y_start: int,
    canvas_width: int,
    line_height: int,
    color: str = "white",
) -> int:
    """Draw lines of text centered horizontally. Returns y after last line."""
    y = y_start
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = (canvas_width - w) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += line_height
    return y


def _prepare_photo(image_path: str) -> Image.Image:
    """
    Load image and scale to 100% video width (1080px), maintaining aspect ratio.
    """
    src = Image.open(image_path).convert("RGB")
    scale = VIDEO_WIDTH / src.width
    new_height = int(src.height * scale)
    src = src.resize((VIDEO_WIDTH, new_height), Image.LANCZOS)
    return src


def _render_text_strip(
    text: str,
    text_viewport_h: int,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
) -> Image.Image:
    """
    Render text onto a tall image (VIDEO_WIDTH x N) with a black background.

    Layout (top to bottom):
      1. Leading blank = text_viewport_h  (text starts off-screen)
      2. TEXT_GAP
      3. Hook text (bold)
      4. Body paragraphs
      5. Trailing blank = 85% of text_viewport_h
         (so last line stops 15% above the viewport bottom)
    """
    max_text_width = VIDEO_WIDTH - 2 * PADDING_X

    # Split text into hook (first paragraph) and body (rest)
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    hook_text = paragraphs[0]
    body_paragraphs = paragraphs[1:]

    # Load fonts
    hook_font = _load_font(bold=True, size=hook_font_size)
    body_font = _load_font(bold=False, size=body_font_size)
    hook_line_h = int(hook_font_size * LINE_SPACING_FACTOR)
    body_line_h = int(body_font_size * LINE_SPACING_FACTOR)

    # Wrap text
    hook_lines = _wrap_text(hook_text, hook_font, max_text_width)
    body_line_groups: list[list[str]] = []
    for para in body_paragraphs:
        body_line_groups.append(_wrap_text(para, body_font, max_text_width))

    for para in body_paragraphs:
        body_line_groups.append(_wrap_text(para, body_font, max_text_width))
        
    # text_viewport_h is now passed in

    # measure raw content height
    # Leading blank removed so text starts visible
    content_h = TEXT_GAP
    content_h += len(hook_lines) * hook_line_h
    content_h += PARAGRAPH_GAP
    for group in body_line_groups:
        content_h += len(group) * body_line_h
        content_h += PARAGRAPH_GAP

    # Trailing blank: just the margin we want (e.g. 20% of viewport)
    trailing = int(text_viewport_h * FOOTER_MARGIN)

    strip_height = content_h + trailing
    
    # Ensure strip is at least as tall as the viewport (plus 1 to be safe)
    # otherwise ffmpeg crop will fail if text is very short
    if strip_height < text_viewport_h:
        strip_height = text_viewport_h + 1

    canvas = Image.new("RGB", (VIDEO_WIDTH, strip_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = TEXT_GAP

    # Hook
    y = _draw_centered_lines(draw, hook_lines, hook_font, y, VIDEO_WIDTH, hook_line_h)
    y += PARAGRAPH_GAP

    # Body
    for group in body_line_groups:
        y = _draw_centered_lines(draw, group, body_font, y, VIDEO_WIDTH, body_line_h)
        y += PARAGRAPH_GAP

    return canvas


def compose_tall_image(
    image_path: str,
    text: str,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
) -> Image.Image:
    """
    Create a tall composite image (kept for compatibility):
      - Source photo at top
      - Text below on black
    """
    photo = _prepare_photo(image_path)
    text_viewport_h = max(0, VIDEO_HEIGHT - photo.height)
    text_strip = _render_text_strip(text, text_viewport_h, hook_font_size, body_font_size)
    image_height = photo.height

    total_height = image_height + text_strip.height
    canvas = Image.new("RGB", (VIDEO_WIDTH, total_height), (0, 0, 0))
    canvas.paste(photo, (0, 0))
    canvas.paste(text_strip, (0, image_height))
    return canvas


def render_video(
    image_path: str,
    text: str,
    output_path: str,
    music_path: str = None,
    music_volume: float = 1.0,
    vo_path: str = None,
    vo_volume: float = 1.0,
    scroll_speed: int = SCROLL_SPEED,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
    fps: int = FPS,
) -> str:
    """
    Render a scrolling video from image + text.

    The photo stays fixed at the top of the viewport.  Only the text
    scrolls upward in the area below the photo.

    Pipeline:
      input 0  – static background (photo top, black bottom, 1080×1920)
      input 1  – tall text strip (scrolled via crop filter)
      overlay  – text strip placed in the lower portion of the frame
    """
    image_height = int(VIDEO_HEIGHT * IMAGE_HEIGHT_RATIO)
    text_viewport_h = VIDEO_HEIGHT - image_height

    # --- Build the two layer images ---
    photo = _prepare_photo(image_path)
    
    # RECALCULATE Heights based on actual photo
    image_height = photo.height
    text_viewport_h = max(1, VIDEO_HEIGHT - image_height) # ensure at least 1px to avoid ffmpeg errors
    
    text_strip = _render_text_strip(text, text_viewport_h, hook_font_size, body_font_size)

    temp_dir = Path(output_path).parent
    stem = Path(output_path).stem
    temp_output = temp_dir / f"{stem}.tmp.mp4"

    # Static background: photo on top, black below (full viewport size)
    bg = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0))
    bg.paste(photo, (0, 0))
    temp_bg = temp_dir / f"_bg_{stem}.png"
    bg.save(str(temp_bg), "PNG")

    # Text strip (only text, no photo)
    temp_txt = temp_dir / f"_txt_{stem}.png"
    text_strip.save(str(temp_txt), "PNG")

    # --- Scroll maths ---
    # Max scroll = text strip height - visible text area
    max_scroll = text_strip.height - text_viewport_h
    if max_scroll <= 0:
        max_scroll = 1

    has_music = bool(music_path) and Path(music_path).exists()
    has_vo = bool(vo_path) and Path(vo_path).exists()

    if has_vo:
        # Sync scroll to the voiceover: the text scrolls exactly as fast as
        # the voice reads it — both start after START_DELAY and end together.
        vo_duration = max(probe_duration(vo_path), 0.1)
        scroll_speed = max_scroll / vo_duration
        scroll_duration = vo_duration
    else:
        scroll_duration = max_scroll / scroll_speed

    # Total duration = start delay + scroll time + hold at end
    duration = START_DELAY + scroll_duration + HOLD_DURATION

    # --- ffmpeg command ---
    # input 0: static background (looped)
    # input 1: tall text strip  (looped)
    # Crop the text strip to text_viewport_h, scrolling y from 0→max_scroll
    # Then overlay it at y=image_height on the static background
    filter_parts = [
        f"[1:v]crop={VIDEO_WIDTH}:{text_viewport_h}:0:"
        f"'min({max_scroll},max(0,t-{START_DELAY})*{scroll_speed:.6f})'[txt];"
        f"[0:v][txt]overlay=0:{image_height}[v_out]"
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1", "-i", str(temp_bg),
        "-loop", "1", "-i", str(temp_txt),
    ]

    audio_labels = []
    input_idx = 2
    if has_music:
        ffmpeg_cmd.extend(["-stream_loop", "-1", "-i", str(music_path)])
        filter_parts.append(f"[{input_idx}:a]volume={music_volume:.3f}[a_m]")
        audio_labels.append("[a_m]")
        input_idx += 1
    if has_vo:
        ffmpeg_cmd.extend(["-i", str(vo_path)])
        vo_delay_ms = int(START_DELAY * 1000)
        filter_parts.append(
            f"[{input_idx}:a]adelay={vo_delay_ms}:all=1,volume={vo_volume:.3f}[a_v]"
        )
        audio_labels.append("[a_v]")
        input_idx += 1

    if len(audio_labels) == 2:
        filter_parts.append(f"{audio_labels[0]}{audio_labels[1]}amix=inputs=2:duration=longest:normalize=0[a_out]")
        audio_map = "[a_out]"
    elif len(audio_labels) == 1:
        audio_map = audio_labels[0]
    else:
        audio_map = None

    ffmpeg_cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[v_out]",
    ])

    if audio_map:
        ffmpeg_cmd.extend(["-map", audio_map, "-c:a", "aac"])

    ffmpeg_cmd.extend([
        "-t", str(duration),
        "-c:v", "libx264",
        "-threads", "4",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_output),
    ])

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

        # Atomic move: rename temp to final
        if Path(output_path).exists():
             Path(output_path).unlink()
        temp_output.rename(output_path)

    finally:
        # Clean up temp files
        for f in (temp_bg, temp_txt, temp_output):
            if f.exists():
                f.unlink()

    return output_path


def batch_render(
    image_paths: list[str],
    texts: list[dict],
    output_dir: str,
    scroll_speed: int = SCROLL_SPEED,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
    progress_callback=None,
) -> list[str]:
    """
    Generate all image × text combinations.

    Args:
        image_paths: List of image file paths
        texts: List of dicts with 'text_id' and 'text' keys
        output_dir: Directory to save videos
        progress_callback: Optional fn(current, total, filename) for progress

    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: list[str] = []
    total = len(image_paths) * len(texts)
    current = 0

    for img_path in image_paths:
        img_name = Path(img_path).stem
        for text_entry in texts:
            text_id = text_entry.get("text_id", "0")
            text = text_entry.get("text", "")
            if not text.strip():
                current += 1
                continue

            out_name = f"{img_name}_text{text_id}.mp4"
            out_path = os.path.join(output_dir, out_name)

            current += 1
            if progress_callback:
                progress_callback(current, total, out_name)

            render_video(
                image_path=img_path,
                text=text,
                output_path=out_path,
                scroll_speed=scroll_speed,
                hook_font_size=hook_font_size,
                body_font_size=body_font_size,
            )
            outputs.append(out_path)

    return outputs
