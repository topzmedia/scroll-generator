"""
Flask web app – Video library with Image Gallery, Text Library & Combination Matrix.
Uses SQLite for persistent storage (replaces library.json / status.json).
"""

import csv
import io
import json
import sqlite3
import threading
import uuid
import zipfile
from pathlib import Path

from PIL import Image as PILImage
import os

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)
from flask_cors import CORS

from renderer import render_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# Allow cross-origin requests from the dashboard
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

BASE_DIR = Path(__file__).parent
LIBRARY_DIR = BASE_DIR / "library"
IMAGES_DIR = LIBRARY_DIR / "images"
VIDEOS_DIR = LIBRARY_DIR / "videos"
UPLOADED_DIR = VIDEOS_DIR / "uploaded"
AUDIO_PATH = BASE_DIR / "audio.mp3"
DB_PATH = LIBRARY_DIR / "library.db"

# Limit concurrent ffmpeg processes to avoid overwhelming the machine
VIDEO_SEMAPHORE = threading.Semaphore(1)

# Legacy JSON paths (for migration only)
_LEGACY_LIBRARY_JSON = LIBRARY_DIR / "library.json"
_LEGACY_STATUS_JSON = LIBRARY_DIR / "status.json"

# ── SQLite helpers ───────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create tables and migrate from legacy JSON files if they exist."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_DIR.mkdir(parents=True, exist_ok=True)

    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT NOT NULL
        );
CREATE TABLE IF NOT EXISTS texts (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gen_status (
            cell_key TEXT PRIMARY KEY,
            status TEXT NOT NULL
        );
    """)
    conn.commit()

    # Add phash column if missing (schema migration)
    try:
        conn.execute("ALTER TABLE images ADD COLUMN phash TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Compute perceptual hashes for images that don't have one yet
    unhashed = conn.execute("SELECT id, path FROM images WHERE phash IS NULL").fetchall()
    if unhashed:
        hashed_count = 0
        for row in unhashed:
            if Path(row["path"]).is_file():
                try:
                    phash = _compute_dhash(row["path"])
                    conn.execute("UPDATE images SET phash = ? WHERE id = ?", (phash, row["id"]))
                    hashed_count += 1
                except Exception as e:
                    print(f"Failed to hash {row['id']}: {e}")
        conn.commit()
        if hashed_count:
            print(f"Computed phash for {hashed_count} image(s)")

    # Reset stale generating video statuses
    reset2 = conn.execute(
        "UPDATE gen_status SET status = 'error' WHERE status = 'generating'"
    )
    if reset2.rowcount:
        conn.commit()
        print(f"Reset {reset2.rowcount} stale video generation(s) to error")

    # Migrate from legacy library.json
    if _LEGACY_LIBRARY_JSON.is_file():
        _migrate_library_json(conn)

    # Migrate from legacy status.json
    if _LEGACY_STATUS_JSON.is_file():
        _migrate_status_json(conn)

    conn.close()


def _migrate_library_json(conn: sqlite3.Connection):
    """Migrate data from library.json (handles corruption) into SQLite."""
    try:
        with open(_LEGACY_LIBRARY_JSON) as f:
            content = f.read()

        # Try normal parse first
        try:
            lib = json.loads(content)
        except json.JSONDecodeError:
            # Corrupted — extract first valid JSON object
            depth = 0
            end = 0
            for i, c in enumerate(content):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                if depth == 0 and i > 0:
                    end = i + 1
                    break
            lib = json.loads(content[:end])
            print(f"Recovered corrupted library.json (kept {end}/{len(content)} chars)")

        # Import images
        for img in lib.get("images", []):
            conn.execute(
                "INSERT OR IGNORE INTO images (id, filename, path) VALUES (?, ?, ?)",
                (img["id"], img["filename"], img["path"]),
            )

        # Import texts
        for txt in lib.get("texts", []):
            conn.execute(
                "INSERT OR IGNORE INTO texts (id, label, text) VALUES (?, ?, ?)",
                (txt["id"], txt["label"], txt["text"]),
            )

        # Import settings
        for k, v in lib.get("settings", {}).items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (k, str(v)),
            )

        conn.commit()
        _LEGACY_LIBRARY_JSON.rename(_LEGACY_LIBRARY_JSON.with_suffix(".json.bak"))
        print(f"Migrated library.json to SQLite ({len(lib.get('images', []))} images, {len(lib.get('texts', []))} texts)")

    except Exception as e:
        print(f"Failed to migrate library.json: {e}")


def _migrate_status_json(conn: sqlite3.Connection):
    """Migrate gen_status from status.json into SQLite."""
    try:
        with open(_LEGACY_STATUS_JSON) as f:
            status = json.load(f)

        for key, value in status.items():
            conn.execute(
                "INSERT OR IGNORE INTO gen_status (cell_key, status) VALUES (?, ?)",
                (key, value),
            )

        conn.commit()
        _LEGACY_STATUS_JSON.rename(_LEGACY_STATUS_JSON.with_suffix(".json.bak"))
        print(f"Migrated status.json to SQLite ({len(status)} entries)")

    except Exception as e:
        print(f"Failed to migrate status.json: {e}")


# ── Perceptual hashing ───────────────────────────────────────────────────────

DHASH_SIZE = 8
DHASH_THRESHOLD = 10  # Hamming distance <= this = duplicate


def _compute_dhash(image_path: str) -> str:
    """Compute dhash (difference hash) for an image. Returns 16-char hex string."""
    img = PILImage.open(image_path).convert("L").resize((DHASH_SIZE + 1, DHASH_SIZE))
    pixels = list(img.getdata())
    bits = []
    for row in range(DHASH_SIZE):
        for col in range(DHASH_SIZE):
            idx = row * (DHASH_SIZE + 1) + col
            bits.append(1 if pixels[idx] < pixels[idx + 1] else 0)
    return format(int("".join(str(b) for b in bits), 2), "016x")


def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _find_duplicate_groups(images: list[dict]) -> dict[str, int | None]:
    """Returns {image_id: group_number_or_None}. Uses union-find to group similar images."""
    hashed = [(img["id"], img["phash"]) for img in images if img.get("phash")]
    parent = {img_id: img_id for img_id, _ in hashed}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            if _hamming_distance(hashed[i][1], hashed[j][1]) <= DHASH_THRESHOLD:
                union(hashed[i][0], hashed[j][0])

    # Assign group numbers to groups with 2+ members
    groups_by_root = {}
    for img_id, _ in hashed:
        root = find(img_id)
        groups_by_root.setdefault(root, []).append(img_id)

    result = {}
    group_num = 1
    for root, members in groups_by_root.items():
        if len(members) >= 2:
            for m in members:
                result[m] = group_num
            group_num += 1

    return result


# ── Database access functions ────────────────────────────────────────────────

def db_get_images() -> list[dict]:
    conn = _get_db()
    rows = conn.execute("SELECT id, filename, path, phash FROM images").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_image(image_id: str) -> dict | None:
    conn = _get_db()
    row = conn.execute("SELECT id, filename, path, phash FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_add_image(img_id: str, filename: str, path: str, phash: str | None = None):
    conn = _get_db()
    with conn:
        conn.execute("INSERT INTO images (id, filename, path, phash) VALUES (?, ?, ?, ?)", (img_id, filename, path, phash))
    conn.close()


def db_delete_image(image_id: str):
    conn = _get_db()
    with conn:
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        # Also clean up gen_status entries for this image
        conn.execute("DELETE FROM gen_status WHERE cell_key LIKE ?", (f"{image_id}__%",))
    conn.close()


def db_get_texts() -> list[dict]:
    conn = _get_db()
    rows = conn.execute("SELECT id, label, text FROM texts").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_text_count() -> int:
    conn = _get_db()
    row = conn.execute("SELECT COUNT(*) FROM texts").fetchone()
    conn.close()
    return row[0]


def db_add_text(text_id: str, label: str, text: str):
    conn = _get_db()
    with conn:
        conn.execute("INSERT INTO texts (id, label, text) VALUES (?, ?, ?)", (text_id, label, text))
    conn.close()


def db_update_text(text_id: str, label: str, text: str):
    conn = _get_db()
    with conn:
        conn.execute("UPDATE texts SET label = ?, text = ? WHERE id = ?", (label, text, text_id))
    conn.close()


def db_delete_text(text_id: str):
    conn = _get_db()
    with conn:
        conn.execute("DELETE FROM texts WHERE id = ?", (text_id,))
        conn.execute("DELETE FROM gen_status WHERE cell_key LIKE ?", (f"%__{text_id}",))
    conn.close()


def db_get_settings() -> dict:
    conn = _get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: int(r["value"]) if r["value"].isdigit() else r["value"] for r in rows}


def db_save_settings(settings: dict):
    conn = _get_db()
    with conn:
        for k, v in settings.items():
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
                (k, str(v), str(v)),
            )
    conn.close()


def db_get_gen_status(key: str) -> str | None:
    conn = _get_db()
    row = conn.execute("SELECT status FROM gen_status WHERE cell_key = ?", (key,)).fetchone()
    conn.close()
    return row["status"] if row else None


def db_get_all_gen_status() -> dict:
    conn = _get_db()
    rows = conn.execute("SELECT cell_key, status FROM gen_status").fetchall()
    conn.close()
    return {r["cell_key"]: r["status"] for r in rows}


def db_set_gen_status(key: str, value: str):
    conn = _get_db()
    with conn:
        conn.execute(
            "INSERT INTO gen_status (cell_key, status) VALUES (?, ?) ON CONFLICT(cell_key) DO UPDATE SET status = ?",
            (key, value, value),
        )
    conn.close()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Images ───────────────────────────────────────────────────────────────────

@app.route("/api/images", methods=["GET"])
def list_images():
    images = db_get_images()
    dup_groups = _find_duplicate_groups(images)

    result = []
    for img in images:
        p = Path(img["path"])
        entry = {
            "id": img["id"],
            "filename": img["filename"],
            "url": f"/api/images/{img['id']}/file",
            "exists": p.is_file(),
            "duplicate_group": dup_groups.get(img["id"]),
        }
        result.append(entry)

    # Sort: duplicates first (grouped), then unique images
    result.sort(key=lambda x: (0 if x["duplicate_group"] else 1, x["duplicate_group"] or 0))
    return jsonify(result)


@app.route("/api/images/<image_id>/file")
def serve_image(image_id):
    img = db_get_image(image_id)
    if not img:
        return jsonify({"error": "Not found"}), 404
    p = Path(img["path"])
    if p.is_file():
        return send_file(str(p))
    return jsonify({"error": "File missing"}), 404


@app.route("/api/images", methods=["POST"])
def upload_images():
    uploaded = []
    files = request.files.getlist("images")
    for f in files:
        if not f.filename:
            continue
        img_id = "img_" + str(uuid.uuid4())[:8]
        ext = Path(f.filename).suffix.lower() or ".png"
        safe_name = f"{img_id}{ext}"
        save_path = IMAGES_DIR / safe_name
        f.save(str(save_path))
        phash = None
        try:
            phash = _compute_dhash(str(save_path))
        except Exception as e:
            print(f"Failed to hash uploaded image {img_id}: {e}")
        db_add_image(img_id, f.filename, str(save_path), phash)
        uploaded.append({"id": img_id, "filename": f.filename, "path": str(save_path)})

    return jsonify({"uploaded": len(uploaded), "images": uploaded})


@app.route("/api/images/<image_id>", methods=["DELETE"])
def delete_image(image_id):
    img = db_get_image(image_id)
    if not img:
        return jsonify({"error": "Not found"}), 404

    # Remove original file
    p = Path(img["path"])
    if p.is_file():
        p.unlink()

    # Remove generated videos for this image
    _remove_videos_for_image(image_id)

    db_delete_image(image_id)
    return jsonify({"message": "Deleted"})


def _remove_videos_for_image(image_id: str):
    """Delete all generated videos that use this image."""
    prefix = f"{image_id}__"
    for vf in VIDEOS_DIR.glob(f"{prefix}*.mp4"):
        vf.unlink()


# ── Texts ────────────────────────────────────────────────────────────────────

@app.route("/api/texts", methods=["GET"])
def list_texts():
    return jsonify(db_get_texts())


@app.route("/api/texts", methods=["POST"])
def add_text():
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Text is required"}), 400

    text_id = "txt_" + str(uuid.uuid4())[:8]
    count = db_get_text_count()
    label = data.get("label") or f"Text {count + 1}"
    text = data["text"].strip()

    db_add_text(text_id, label, text)
    entry = {"id": text_id, "label": label, "text": text}
    return jsonify(entry)


@app.route("/api/texts/import", methods=["POST"])
def import_texts():
    csv_file = request.files.get("csv_file")
    if not csv_file:
        return jsonify({"error": "No CSV file"}), 400

    content = csv_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    added = []
    count = db_get_text_count()
    for row in reader:
        text = row.get("text", "")
        if not text.strip():
            continue
        text = text.replace("\\n", "\n")
        text_id = "txt_" + str(uuid.uuid4())[:8]
        count += 1
        label = row.get("label", row.get("text_id", f"Text {count}"))
        db_add_text(text_id, str(label), text.strip())
        added.append({"id": text_id, "label": str(label), "text": text.strip()})

    return jsonify({"imported": len(added), "texts": added})


@app.route("/api/texts/<text_id>", methods=["PUT", "DELETE"])
def manage_text(text_id):
    if request.method == "PUT":
        data = request.get_json()
        if not data or not data.get("text", "").strip():
            return jsonify({"error": "Text is required"}), 400

        texts = db_get_texts()
        old = next((t for t in texts if t["id"] == text_id), None)
        if not old:
            return jsonify({"error": "Not found"}), 404

        new_label = (data.get("label") or old["label"]).strip()
        new_text = data["text"].strip()

        # If text content changed, delete stale videos
        if new_text != old["text"]:
            _remove_videos_for_text(text_id)
            conn = _get_db()
            with conn:
                conn.execute("DELETE FROM gen_status WHERE cell_key LIKE ?", (f"%__{text_id}",))
            conn.close()

        db_update_text(text_id, new_label, new_text)
        return jsonify({"id": text_id, "label": new_label, "text": new_text})

    # DELETE
    texts = db_get_texts()
    found = any(t["id"] == text_id for t in texts)
    if not found:
        return jsonify({"error": "Not found"}), 404

    _remove_videos_for_text(text_id)
    db_delete_text(text_id)
    return jsonify({"message": "Deleted"})


def _remove_videos_for_text(text_id: str):
    """Delete all generated videos that use this text."""
    suffix = f"__{text_id}.mp4"
    for vf in VIDEOS_DIR.glob(f"*{suffix}"):
        vf.unlink()


# ── Matrix / Library state ───────────────────────────────────────────────────

@app.route("/api/library")
def get_library():
    """Return full library state including which videos exist."""
    images = db_get_images()
    texts = db_get_texts()
    settings = db_get_settings()
    all_status = db_get_all_gen_status()

    # Build video status map
    videos = {}
    for img in images:
        for txt in texts:
            cell_key = f"{img['id']}__{txt['id']}"
            video_path = VIDEOS_DIR / f"{cell_key}.mp4"
            gs = all_status.get(cell_key)
            if gs == "generating":
                videos[cell_key] = {"status": "generating"}
            elif gs == "uploaded":
                uploaded_path = UPLOADED_DIR / f"{cell_key}.mp4"
                if uploaded_path.is_file():
                    stat = uploaded_path.stat()
                    videos[cell_key] = {
                        "status": "uploaded",
                        "size": stat.st_size,
                        "url": f"/api/video/{cell_key}",
                    }
            elif video_path.is_file():
                stat = video_path.stat()
                videos[cell_key] = {
                    "status": "done",
                    "size": stat.st_size,
                    "url": f"/api/video/{cell_key}",
                }

    # Build image list
    image_list = []
    for img in images:
        image_list.append({
            "id": img["id"],
            "filename": img["filename"],
            "url": f"/api/images/{img['id']}/file",
        })

    return jsonify({
        "images": image_list,
        "texts": texts,
        "videos": videos,
        "settings": settings,
    })


# ── Video serving ────────────────────────────────────────────────────────────

@app.route("/api/video/<cell_key>")
def serve_video(cell_key):
    safe = Path(cell_key).name
    video_path = VIDEOS_DIR / f"{safe}.mp4"
    if video_path.is_file():
        return send_file(str(video_path), mimetype="video/mp4")
    uploaded_path = UPLOADED_DIR / f"{safe}.mp4"
    if uploaded_path.is_file():
        return send_file(str(uploaded_path), mimetype="video/mp4")
    return jsonify({"error": "Not found"}), 404


@app.route("/api/video/<cell_key>/upload", methods=["POST"])
def mark_video_uploaded(cell_key):
    safe = Path(cell_key).name
    video_path = VIDEOS_DIR / f"{safe}.mp4"
    uploaded_path = UPLOADED_DIR / f"{safe}.mp4"
    if not video_path.is_file():
        return jsonify({"error": "Video not found"}), 404
    video_path.rename(uploaded_path)
    db_set_gen_status(safe, "uploaded")
    return jsonify({"status": "uploaded"})


@app.route("/api/video/<cell_key>", methods=["DELETE"])
def delete_video(cell_key):
    safe = Path(cell_key).name
    video_path = VIDEOS_DIR / f"{safe}.mp4"
    uploaded_path = UPLOADED_DIR / f"{safe}.mp4"
    if video_path.is_file():
        video_path.unlink()
    if uploaded_path.is_file():
        uploaded_path.unlink()
    db_set_gen_status(safe, "error")
    return jsonify({"message": "Deleted"})


# ── Generate ─────────────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate one or more videos.  Body: {pairs: [{image_id, text_id}, ...], settings: {}}"""
    data = request.get_json()
    pairs = data.get("pairs", [])
    settings = data.get("settings", {})

    if not pairs:
        return jsonify({"error": "No pairs specified"}), 400

    images = db_get_images()
    texts = db_get_texts()
    img_map = {i["id"]: i for i in images}
    txt_map = {t["id"]: t for t in texts}

    scroll_speed = int(settings.get("scroll_speed", 80))
    hook_font_size = int(settings.get("hook_font_size", 52))
    body_font_size = int(settings.get("body_font_size", 38))

    # Save settings
    db_save_settings({
        "scroll_speed": scroll_speed,
        "hook_font_size": hook_font_size,
        "body_font_size": body_font_size,
    })

    valid_pairs = []
    for p in pairs:
        img = img_map.get(p["image_id"])
        txt = txt_map.get(p["text_id"])
        if img and txt and Path(img["path"]).is_file():
            cell_key = f"{img['id']}__{txt['id']}"
            db_set_gen_status(cell_key, "generating")
            valid_pairs.append((img, txt, cell_key))

    if not valid_pairs:
        return jsonify({"error": "No valid pairs"}), 400

    def _run():
        for img, txt, cell_key in valid_pairs:
            # Delete old video so stale file doesn't persist on failure
            old_video = VIDEOS_DIR / f"{cell_key}.mp4"
            if old_video.is_file():
                old_video.unlink()

            actual_path = img["path"]

            out_path = str(VIDEOS_DIR / f"{cell_key}.mp4")
            VIDEO_SEMAPHORE.acquire()
            try:
                render_video(
                    image_path=actual_path,
                    text=txt["text"],
                    output_path=out_path,
                    scroll_speed=scroll_speed,
                    hook_font_size=hook_font_size,
                    body_font_size=body_font_size,
                    audio_path=str(AUDIO_PATH),
                )
                db_set_gen_status(cell_key, "done")
            except Exception as e:
                db_set_gen_status(cell_key, "error")
                print(f"Error generating {cell_key}: {e}")
            finally:
                VIDEO_SEMAPHORE.release()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({
        "message": f"Generating {len(valid_pairs)} video(s)",
        "cells": [p[2] for p in valid_pairs],
    })


@app.route("/api/generate/status")
def generation_status():
    """Return current generation statuses for all cells."""
    return jsonify(db_get_all_gen_status())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update global settings."""
    new_settings = request.get_json() or {}
    valid_keys = {"scroll_speed", "hook_font_size", "body_font_size"}

    to_save = {}
    for k in valid_keys:
        if k in new_settings:
            try:
                to_save[k] = int(new_settings[k])
            except (ValueError, TypeError):
                pass

    if to_save:
        db_save_settings(to_save)

    return jsonify(db_get_settings())


# ── Download ZIP ─────────────────────────────────────────────────────────────

@app.route("/api/download", methods=["POST"])
def download():
    """Download selected videos as ZIP.  Body: {cells: [cell_key, ...]}"""
    data = request.get_json() or {}
    cells = data.get("cells", [])

    if not cells:
        cells = [f.stem for f in VIDEOS_DIR.glob("*.mp4")]

    if not cells:
        return jsonify({"error": "No videos to download"}), 404

    zip_path = LIBRARY_DIR / "download.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for cell_key in cells:
            vp = VIDEOS_DIR / f"{cell_key}.mp4"
            up = UPLOADED_DIR / f"{cell_key}.mp4"
            if vp.is_file():
                zf.write(str(vp), vp.name)
            elif up.is_file():
                zf.write(str(up), up.name)

    return send_file(
        str(zip_path),
        mimetype="application/zip",
        as_attachment=True,
        download_name="videos.zip",
    )


# ── Init ─────────────────────────────────────────────────────────────────────

_init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=True)
