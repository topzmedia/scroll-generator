"""
Video Renderer — creates scrolling text videos from image + text.

Pipeline:
1. Pillow composites a tall image: photo on top → text below on black
2. ffmpeg pans down over it to produce a 1080×1920 MP4
"""

import os
import re
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

# Poppins has no emoji glyphs, so emoji are drawn from a colour emoji font.
# Noto Color Emoji is a CBDT bitmap font: FreeType only accepts its native
# strike size, so every emoji is rendered at EMOJI_STRIKE and scaled down.
EMOJI_FONT_CANDIDATES = [
    FONT_DIR / "NotoColorEmoji.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    Path("/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf"),
]
EMOJI_STRIKE = 109

# Emoji clusters: flags (regional-indicator pairs), keycaps, and any emoji base
# plus its variation selectors / skin tones / ZWJ continuations.
_EMOJI_BASE = (
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U0001F000-\U0001F0FF"
    "\U0001F170-\U0001F251☀-➿⬀-⯿←-⇿"
    "⌀-⏿■-◿⁉‼™ℹ"
)
_MODIFIER = "[️︎\U0001F3FB-\U0001F3FF]*"
_ATOM = f"[{_EMOJI_BASE}]{_MODIFIER}"
EMOJI_RE = re.compile(
    "(?:"
    "[\U0001F1E6-\U0001F1FF]{2}"          # flags
    "|[0-9#*]️?⃣"               # keycaps
    f"|{_ATOM}(?:‍{_ATOM})*"         # emoji, incl. ZWJ sequences
    ")"
)

_emoji_font = None
_emoji_font_loaded = False
_emoji_cache: dict = {}


def _load_emoji_font():
    """Load the colour emoji font once; None when unavailable."""
    global _emoji_font, _emoji_font_loaded
    if _emoji_font_loaded:
        return _emoji_font
    _emoji_font_loaded = True
    for path in EMOJI_FONT_CANDIDATES:
        if path.exists():
            try:
                _emoji_font = ImageFont.truetype(str(path), EMOJI_STRIKE)
                break
            except OSError as e:
                print(f"Emoji font {path} unusable: {e}")
    if _emoji_font is None:
        print("No colour emoji font found — emoji will be dropped from the text")
    return _emoji_font


def _render_emoji(cluster: str, target_h: int) -> Image.Image | None:
    """Return the emoji drawn at target_h pixels tall (cached), or None."""
    key = (cluster, target_h)
    if key in _emoji_cache:
        return _emoji_cache[key]

    font = _load_emoji_font()
    img = None
    if font is not None:
        try:
            canvas = Image.new("RGBA", (EMOJI_STRIKE * 3, EMOJI_STRIKE * 2), (0, 0, 0, 0))
            ImageDraw.Draw(canvas).text((EMOJI_STRIKE // 4, EMOJI_STRIKE // 4), cluster,
                                        font=font, embedded_color=True)
            bbox = canvas.getbbox()
            if bbox:
                glyph = canvas.crop(bbox)
                scale = target_h / glyph.height
                img = glyph.resize((max(1, round(glyph.width * scale)), target_h), Image.LANCZOS)
        except Exception as e:
            print(f"Failed to render emoji {cluster!r}: {e}")
    _emoji_cache[key] = img
    return img


def _emoji_metrics(font_size: int) -> tuple[int, int]:
    """(target height, trailing gap) for emoji drawn alongside font_size text."""
    return max(8, round(font_size * 1.05)), max(2, round(font_size * 0.08))


def _split_runs(text: str) -> list[tuple[str, str]]:
    """Split text into ('text'|'emoji', chunk) runs in order."""
    runs = []
    pos = 0
    for m in EMOJI_RE.finditer(text):
        if m.start() > pos:
            runs.append(("text", text[pos:m.start()]))
        runs.append(("emoji", m.group()))
        pos = m.end()
    if pos < len(text):
        runs.append(("text", text[pos:]))
    return runs


def measure_text(text: str, font: ImageFont.FreeTypeFont) -> float:
    """Advance width of text, counting emoji at their scaled bitmap width."""
    emoji_h, emoji_gap = _emoji_metrics(font.size)
    total = 0.0
    for kind, chunk in _split_runs(text):
        if kind == "text":
            total += font.getlength(chunk)
        else:
            img = _render_emoji(chunk, emoji_h)
            total += (img.width + emoji_gap) if img else 0
    return total


def _draw_rich_text(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    x: float,
    y: int,
    color: str,
) -> float:
    """
    Draw text at (x, y) with emoji pulled from the colour emoji font.

    Returns the x position after the last run.  Emoji are centred on the text's
    own visual centre so they sit on the same line as the words.
    """
    emoji_h, emoji_gap = _emoji_metrics(font.size)
    ascent, _ = font.getmetrics()
    centre_y = y + ascent - round(font.size * 0.35)

    for kind, chunk in _split_runs(text):
        if kind == "text":
            draw.text((x, y), chunk, font=font, fill=color)
            x += font.getlength(chunk)
        else:
            img = _render_emoji(chunk, emoji_h)
            if img:
                canvas.alpha_composite(img, (round(x), centre_y - img.height // 2)) \
                    if canvas.mode == "RGBA" else canvas.paste(img, (round(x), centre_y - img.height // 2), img)
                x += img.width + emoji_gap
    return x


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
        w = measure_text(test_line, font)
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
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    y_start: int,
    canvas_width: int,
    line_height: int,
    color: str = "white",
    bold: bool = False,
    word_boxes: list[dict] | None = None,
) -> int:
    """
    Draw lines of text centered horizontally. Returns y after last line.

    When word_boxes is given, each word's position is recorded so the renderer
    can place the karaoke highlight exactly over it later.
    """
    y = y_start
    space_w = font.getlength(" ")
    for line in lines:
        x = (canvas_width - measure_text(line, font)) / 2
        wx = x
        for word in line.split(" "):
            if not word:
                wx += space_w
                continue
            word_w = measure_text(word, font)
            _draw_rich_text(canvas, draw, word, font, wx, y, color)
            if word_boxes is not None:
                word_boxes.append({
                    "text": word,
                    "x": int(wx),
                    "y": y,
                    "w": int(word_w),
                    "font_size": font.size,
                    "line_h": line_height,
                    "bold": bold,
                })
            wx += word_w + space_w
        y += line_height
    return y


def _prepare_photo(image_path: str, target_height: int | None = None) -> Image.Image:
    """
    Load image and scale to 100% video width (1080px), maintaining aspect ratio.

    With target_height, the photo instead fills exactly 1080×target_height:
    scaled to cover, then centre-cropped (no distortion, no letterboxing).
    """
    src = Image.open(image_path).convert("RGB")
    if not target_height:
        scale = VIDEO_WIDTH / src.width
        return src.resize((VIDEO_WIDTH, int(src.height * scale)), Image.LANCZOS)

    scale = max(VIDEO_WIDTH / src.width, target_height / src.height)
    new_w, new_h = max(1, round(src.width * scale)), max(1, round(src.height * scale))
    src = src.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - VIDEO_WIDTH) // 2
    top = (new_h - target_height) // 2
    return src.crop((left, top, left + VIDEO_WIDTH, top + target_height))


def _render_text_strip(
    text: str,
    text_viewport_h: int,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
    label: str = "",
    bg_color: str = "#000000",
    text_color: str = "#ffffff",
    word_boxes: list[dict] | None = None,
) -> Image.Image:
    """
    Render text onto a tall image (VIDEO_WIDTH x N) with a black background.

    Layout (top to bottom):
      1. TEXT_GAP
      2. Label (bold, hook_font_size) — from the text entry's label field
      3. PARAGRAPH_GAP
      4. Full text content (regular, body_font_size) — all paragraphs
      5. Trailing blank

    word_boxes, when provided, is filled with each drawn word's position for
    the karaoke highlight overlay.
    """
    max_text_width = VIDEO_WIDTH - 2 * PADDING_X

    # Label = the text entry's label field
    label_text = label.strip() if label else ""

    # Body = the full text content split into paragraphs for rendering
    body_paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not body_paragraphs:
        body_paragraphs = [text.strip()]

    # Load fonts
    label_font = _load_font(bold=True, size=hook_font_size)
    body_font = _load_font(bold=False, size=body_font_size)
    label_line_h = int(hook_font_size * LINE_SPACING_FACTOR)
    body_line_h = int(body_font_size * LINE_SPACING_FACTOR)

    # Wrap text
    label_lines = _wrap_text(label_text, label_font, max_text_width) if label_text else []
    body_line_groups: list[list[str]] = []
    for para in body_paragraphs:
        body_line_groups.append(_wrap_text(para, body_font, max_text_width))

    # Measure content height
    content_h = TEXT_GAP
    if label_lines:
        content_h += len(label_lines) * label_line_h
        content_h += PARAGRAPH_GAP
    for group in body_line_groups:
        content_h += len(group) * body_line_h
        content_h += PARAGRAPH_GAP

    trailing = int(text_viewport_h * FOOTER_MARGIN)
    strip_height = content_h + trailing

    if strip_height < text_viewport_h:
        strip_height = text_viewport_h + 1

    canvas = Image.new("RGB", (VIDEO_WIDTH, strip_height), bg_color)
    draw = ImageDraw.Draw(canvas)
    y = TEXT_GAP

    # Label
    if label_lines:
        y = _draw_centered_lines(canvas, draw, label_lines, label_font, y, VIDEO_WIDTH, label_line_h,
                                 color=text_color, bold=True, word_boxes=word_boxes)
        y += PARAGRAPH_GAP

    # Body
    for group in body_line_groups:
        y = _draw_centered_lines(canvas, draw, group, body_font, y, VIDEO_WIDTH, body_line_h,
                                 color=text_color, bold=False, word_boxes=word_boxes)
        y += PARAGRAPH_GAP

    return canvas


def _render_title_band(
    label_text: str,
    hook_font_size: int,
    color: str,
    bg: str,
) -> Image.Image:
    """
    The sticky title: the text's label on its own band, pinned right below
    the image/video while the body scrolls underneath it.
    """
    max_text_width = VIDEO_WIDTH - 2 * PADDING_X
    font = _load_font(bold=True, size=hook_font_size)
    line_h = int(hook_font_size * LINE_SPACING_FACTOR)
    lines = _wrap_text(label_text, font, max_text_width)
    band = Image.new("RGB", (VIDEO_WIDTH, 2 * TEXT_GAP + line_h * len(lines)), bg)
    draw = ImageDraw.Draw(band)
    _draw_centered_lines(band, draw, lines, font, TEXT_GAP, VIDEO_WIDTH, line_h,
                         color=color, bold=True)
    return band


def _normalize_word(word: str) -> str:
    """Comparable form of a word: lowercase letters and digits only."""
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _drop_label_prefix(spoken: list[dict], label_text: str) -> list[dict]:
    """
    The voice reads the label first, but with a sticky title the label is not
    part of the scrolling strip — drop those leading spoken words so
    alignment starts at the body copy.
    """
    targets = [t for t in (_normalize_word(w) for w in label_text.split()) if t]
    if not targets:
        return spoken
    i, ti = 0, 0
    while i < len(spoken) and ti < len(targets) and i <= len(targets) * 3:
        nw = _normalize_word(spoken[i].get("text", ""))
        if not nw:
            i += 1
            continue
        if nw == targets[ti] or nw.startswith(targets[ti]) or targets[ti].startswith(nw):
            ti += 1
        i += 1
    return spoken[i:]


def _align_words(spoken: list[dict], boxes: list[dict]) -> list[dict]:
    """
    Match TTS word timings to drawn word boxes, in order.

    The two lists usually agree word-for-word, but a TTS engine may split or
    skip tokens (numbers, symbols).  Walking both with a lookahead keeps the
    rest of the sentence aligned instead of drifting after one mismatch.
    """
    matched = []
    bi = 0
    for sw in spoken:
        target = _normalize_word(sw.get("text", ""))
        if not target:
            continue
        for probe in range(bi, min(bi + 6, len(boxes))):
            box = boxes[probe]
            norm = _normalize_word(box["text"])
            if not norm:
                continue
            if norm == target or norm.startswith(target) or target.startswith(norm):
                matched.append({**box, "start": sw["start"], "end": sw["end"]})
                bi = probe + 1
                break
    return matched


def _build_scroll_expr(
    words: list[dict],
    max_scroll: int,
    viewport_h: int,
    start_delay: float,
    scroll_end: float,
) -> str | None:
    """
    Piecewise-linear scroll that follows the speech tempo: at the moment a
    word is spoken, the line it sits on is about 1/3 down the visible text
    area — never above the fold, never near the bottom.  One breakpoint per
    text line (the first spoken word on it); between breakpoints the scroll
    moves linearly, so pauses in the voice pause the scroll too.

    Returns an ffmpeg time expression, or None when the timings give fewer
    than two usable lines (caller falls back to the linear scroll).
    """
    anchor = viewport_h / 3.0
    points = [(0.0, 0.0)]
    last_y = None
    for w in words:
        y, t0 = w.get("y"), w.get("start")
        if y is None or t0 is None:
            continue
        if last_y is not None and y <= last_y:
            continue  # same line — one breakpoint per line is enough
        last_y = y
        t = start_delay + max(0.0, t0)
        target = min(float(max_scroll), max(0.0, y - anchor))
        prev_t, prev_y = points[-1]
        if t <= prev_t + 0.05:
            continue
        points.append((t, max(target, prev_y)))  # never scroll backwards

    if len(points) < 3:
        return None

    # Land on the very end of the strip: by speech end when the timings leave
    # room, otherwise slide the last bit during the start of the hold.
    t_last, y_last = points[-1]
    if y_last < max_scroll:
        t_end = scroll_end if scroll_end > t_last + 0.3 else t_last + min(1.5, HOLD_DURATION / 2)
        points.append((t_end, float(max_scroll)))

    terms = []
    for (t0, y0), (t1, y1) in zip(points, points[1:]):
        slope = (y1 - y0) / (t1 - t0)
        terms.append(
            f"(gte(t,{t0:.4f})*lt(t,{t1:.4f}))*({y0:.2f}+(t-{t0:.4f})*{slope:.4f})"
        )
    t_final, y_final = points[-1]
    terms.append(f"gte(t,{t_final:.4f})*{y_final:.2f}")
    return f"min({max_scroll},max(0,{'+'.join(terms)}))"


def _build_highlight_layer(
    words: list[dict],
    temp_dir: Path,
    stem: str,
    start_delay: float,
    total_duration: float,
    text_color: str,
    bg_color: str,
) -> tuple[Path, str, int] | None:
    """
    Build the karaoke highlight layer.

    Each highlighted word becomes one full-width RGBA patch (transparent
    except for the word and its highlight box), sequenced by a concat file so
    the patch swaps exactly when the voice moves on.  Returns
    (concat_file, y_expression, patch_height) — the y expression tracks the
    highlighted word's position inside the scrolling strip.
    """
    if not words:
        return None

    pad_x, pad_y = 8, 8
    patch_h = max(w.get("line_h") or int(w["font_size"] * LINE_SPACING_FACTOR)
                  for w in words) + 2 * pad_y

    blank = Image.new("RGBA", (VIDEO_WIDTH, patch_h), (0, 0, 0, 0))
    blank_path = temp_dir / f"_hl_{stem}_blank.png"
    blank.save(str(blank_path), "PNG")

    segments = []  # (image_path, duration, y_offset_in_strip)
    cursor = 0.0
    created = [blank_path]

    for idx, word in enumerate(words):
        w_start = start_delay + max(0.0, word["start"])
        w_end = min(start_delay + max(word["end"], word["start"] + 0.05), total_duration)
        if w_end <= w_start:
            continue
        if w_start > cursor:
            segments.append((blank_path, w_start - cursor, 0))

        patch = Image.new("RGBA", (VIDEO_WIDTH, patch_h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(patch)
        font = _load_font(bold=True, size=word["font_size"])
        text_w = measure_text(word["text"], font)
        # Bold is wider than the regular glyphs underneath: grow the box around
        # the original word's centre so the highlight stays centred on it.
        cx = word["x"] + word["w"] / 2
        x0 = cx - text_w / 2
        # Keep the box inside its own line so it never clips the line below.
        line_h = word.get("line_h") or int(word["font_size"] * LINE_SPACING_FACTOR)
        ascent, descent = font.getmetrics()
        pad_v = 3
        box_top = max(0, pad_y - pad_v)
        box_bottom = min(pad_y + line_h - 2, pad_y + ascent + descent + pad_v)
        pdraw.rounded_rectangle(
            [x0 - pad_x, box_top, x0 + text_w + pad_x, box_bottom],
            radius=8, fill=bg_color,
        )
        _draw_rich_text(patch, pdraw, word["text"], font, x0, pad_y, text_color)

        patch_path = temp_dir / f"_hl_{stem}_{idx:04d}.png"
        patch.save(str(patch_path), "PNG")
        created.append(patch_path)
        segments.append((patch_path, w_end - w_start, word["y"] - pad_y))
        cursor = w_end

    if cursor < total_duration:
        segments.append((blank_path, total_duration - cursor, 0))
    if not segments:
        return None

    # concat demuxer script — one entry per segment, last frame repeated so the
    # stream never runs dry before the video ends.
    lines = ["ffconcat version 1.0"]
    for path, duration, _ in segments:
        lines.append(f"file '{path.name}'")
        lines.append(f"duration {duration:.4f}")
    lines.append(f"file '{segments[-1][0].name}'")
    concat_path = temp_dir / f"_hl_{stem}.txt"
    concat_path.write_text("\n".join(lines) + "\n")

    # y(t) as a flat sum of half-open time windows — one term per segment.
    terms = []
    t = 0.0
    for _, duration, y_off in segments:
        t_end = t + duration
        if y_off:
            terms.append(f"{y_off}*(gte(t,{t:.4f})*lt(t,{t_end:.4f}))")
        t = t_end
    y_expr = "+".join(terms) if terms else "0"

    return concat_path, y_expr, patch_h


def compose_tall_image(
    image_path: str,
    text: str,
    hook_font_size: int = HOOK_FONT_SIZE,
    body_font_size: int = BODY_FONT_SIZE,
    label: str = "",
) -> Image.Image:
    """
    Create a tall composite image (kept for compatibility):
      - Source photo at top
      - Text below on black
    """
    photo = _prepare_photo(image_path)
    text_viewport_h = max(0, VIDEO_HEIGHT - photo.height)
    text_strip = _render_text_strip(text, text_viewport_h, hook_font_size, body_font_size, label=label)
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
    label: str = "",
    bg_color: str = "#000000",
    text_color: str = "#ffffff",
    image_height_pct: int = 0,
    vo_words: list[dict] | None = None,
    highlight: bool = False,
    highlight_text_color: str = "#000000",
    highlight_bg_color: str = "#ffd93d",
    title_text_color: str = None,
    title_bg_color: str = None,
) -> str:
    """
    Render a scrolling video from image + text.

    The photo stays fixed at the top of the viewport.  Only the text
    scrolls upward in the area below the photo.

    image_height_pct (1-95) fixes the photo to that share of the frame height,
    leaving the rest for the scrolling text; 0 keeps the photo's own aspect
    ratio.  With highlight and vo_words, the word currently being spoken is
    drawn bold on a coloured highlight box.
    """
    pct = int(image_height_pct or 0)

    # --- Sticky title band: the label pinned right below the image ---
    label_text = label.strip() if label else ""
    title_band = None
    if label_text:
        title_band = _render_title_band(label_text, hook_font_size,
                                        title_text_color or text_color,
                                        title_bg_color or bg_color)
    title_h = title_band.height if title_band else 0
    # The scrolling body must keep a readable viewport below the band.
    MIN_TEXT_VIEWPORT = 240

    # --- Build the two layer images ---
    if pct > 0:
        pct = max(1, min(95, pct))
        image_height = max(1, round(VIDEO_HEIGHT * pct / 100))
        image_height = min(image_height, max(1, VIDEO_HEIGHT - title_h - MIN_TEXT_VIEWPORT))
        photo = _prepare_photo(image_path, target_height=image_height)
        text_viewport_h = max(1, VIDEO_HEIGHT - image_height - title_h)
    else:
        photo = _prepare_photo(image_path)
        image_height = photo.height
        # Shrink the photo if the band + a readable viewport don't fit below.
        max_image_h = max(1, VIDEO_HEIGHT - title_h - MIN_TEXT_VIEWPORT)
        if image_height > max_image_h:
            image_height = max_image_h
            scale = image_height / photo.height
            photo = photo.resize((int(photo.width * scale), image_height), Image.LANCZOS)
            if photo.width > VIDEO_WIDTH:
                left = (photo.width - VIDEO_WIDTH) // 2
                photo = photo.crop((left, 0, left + VIDEO_WIDTH, image_height))
        text_viewport_h = max(1, VIDEO_HEIGHT - image_height - title_h)

    word_boxes: list[dict] = []
    # The label lives on the sticky band now, so the strip is body copy only.
    text_strip = _render_text_strip(text, text_viewport_h, hook_font_size, body_font_size,
                                    label="", bg_color=bg_color, text_color=text_color,
                                    word_boxes=word_boxes)

    temp_dir = Path(output_path).parent
    stem = Path(output_path).stem
    temp_output = temp_dir / f"{stem}.tmp.mp4"

    # Static background: photo on top, colored below (full viewport size)
    bg = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
    bg.paste(photo, (0, 0))
    if title_band is not None:
        bg.paste(title_band, (0, image_height))
    temp_bg = temp_dir / f"_bg_{stem}.png"
    bg.save(str(temp_bg), "PNG")

    # Text strip (only text, no photo)
    temp_txt = temp_dir / f"_txt_{stem}.png"
    text_strip.save(str(temp_txt), "PNG")

    # --- Scroll maths ---
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

    duration = START_DELAY + scroll_duration + HOLD_DURATION

    # --- Word alignment (drives both the speech-synced scroll and karaoke) ---
    aligned_words: list[dict] = []
    if has_vo and vo_words:
        # The voice reads the label first, but the sticky title is not part of
        # the strip — align (and highlight) from the body copy onward.
        spoken = _drop_label_prefix(vo_words, label_text) if title_band else vo_words
        aligned_words = _align_words(spoken, word_boxes)

    # --- Karaoke highlight layer (needs the scroll timing above) ---
    highlight_layer = None
    if highlight and aligned_words:
        highlight_layer = _build_highlight_layer(
            aligned_words, temp_dir, stem,
            START_DELAY, duration, highlight_text_color, highlight_bg_color,
        )

    # --- ffmpeg command ---
    # Whenever there is a voice with word timings, scroll follows the speech
    # so the spoken word stays ~1/3 down the text area; without timings,
    # fall back to the linear scroll.
    scroll_expr = f"min({max_scroll},max(0,t-{START_DELAY})*{scroll_speed:.6f})"
    if aligned_words:
        speech_expr = _build_scroll_expr(
            aligned_words, max_scroll, text_viewport_h,
            START_DELAY, START_DELAY + scroll_duration,
        )
        if speech_expr:
            scroll_expr = speech_expr
    filter_parts = [
        f"[1:v]crop={VIDEO_WIDTH}:{text_viewport_h}:0:'{scroll_expr}'[txt]"
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1", "-i", str(temp_bg),
        "-loop", "1", "-i", str(temp_txt),
    ]

    input_idx = 2
    text_label = "[txt]"
    if highlight_layer:
        concat_path, y_expr, patch_h = highlight_layer
        ffmpeg_cmd.extend(["-f", "concat", "-safe", "0", "-i", str(concat_path)])
        # Overlaying onto the cropped text layer (not the full frame) clips the
        # highlight to the text area, so it can never paint over the photo.
        filter_parts.append(f"[{input_idx}:v]format=rgba[hl]")
        filter_parts.append(
            f"[txt][hl]overlay=x=0:y='({y_expr})-({scroll_expr})':eval=frame:format=auto[txt_hl]"
        )
        text_label = "[txt_hl]"
        input_idx += 1

    filter_parts.append(f"[0:v]{text_label}overlay=0:{image_height + title_h}[v_out]")

    audio_labels = []
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

        if Path(output_path).exists():
             Path(output_path).unlink()
        temp_output.rename(output_path)

    finally:
        temp_files = [temp_bg, temp_txt, temp_output]
        temp_files.extend(temp_dir.glob(f"_hl_{stem}*"))
        for f in temp_files:
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
                label=text_entry.get("label", ""),
            )
            outputs.append(out_path)

    return outputs
