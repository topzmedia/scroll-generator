"""
Flask web app – Video library with Image Gallery, Text Library & Combination Matrix.
Uses SQLite for persistent storage (replaces library.json / status.json).
Uses Cloudflare R2 (S3-compatible) for image/video file storage.
"""

import csv
import io
import json
import sqlite3
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from PIL import Image as PILImage
import os
import boto3
from botocore.exceptions import ClientError

from flask import (
    Flask,
    Response,
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
AUDIO_PATH = BASE_DIR / "audio.mp3"
DB_PATH = LIBRARY_DIR / "library.db"

# Limit concurrent ffmpeg processes to avoid overwhelming the machine
VIDEO_SEMAPHORE = threading.Semaphore(1)

# Legacy JSON paths (for migration only)
_LEGACY_LIBRARY_JSON = LIBRARY_DIR / "library.json"
_LEGACY_STATUS_JSON = LIBRARY_DIR / "status.json"

# ── R2 / S3 configuration ────────────────────────────────────────────────────

R2_ENDPOINT = "https://346c911ecd013de0d51b44f77e5f2ec0.r2.cloudflarestorage.com"
R2_BUCKET = "scroll-generator"

_s3_client = None
_s3_lock = threading.Lock()


def _get_s3():
    """Return a shared boto3 S3 client (created lazily, thread-safe)."""
    global _s3_client
    if _s3_client is None:
        with _s3_lock:
            if _s3_client is None:
                _s3_client = boto3.client(
                    "s3",
                    endpoint_url=R2_ENDPOINT,
                    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                    region_name="auto",
                )
    return _s3_client


def r2_upload_fileobj(fileobj, key: str, content_type: str | None = None):
    """Upload a file-like object to R2."""
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    _get_s3().upload_fileobj(fileobj, R2_BUCKET, key, ExtraArgs=extra or None)


def r2_upload_file(local_path: str, key: str, content_type: str | None = None):
    """Upload a local file to R2."""
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    _get_s3().upload_file(local_path, R2_BUCKET, key, ExtraArgs=extra or None)


def r2_download_file(key: str, local_path: str):
    """Download a file from R2 to a local path."""
    _get_s3().download_file(R2_BUCKET, key, local_path)


def r2_delete(key: str):
    """Delete an object from R2. Silently ignores if it doesn't exist."""
    try:
        _get_s3().delete_object(Bucket=R2_BUCKET, Key=key)
    except ClientError:
        pass


def r2_exists(key: str) -> bool:
    """Check if an object exists in R2."""
    try:
        _get_s3().head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def r2_get_size(key: str) -> int | None:
    """Return the size of an R2 object in bytes, or None if it doesn't exist."""
    try:
        resp = _get_s3().head_object(Bucket=R2_BUCKET, Key=key)
        return resp["ContentLength"]
    except ClientError:
        return None


def r2_get_object(key: str) -> bytes | None:
    """Download an R2 object into memory. Returns None if not found."""
    try:
        resp = _get_s3().get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except ClientError:
        return None


def r2_list_keys(prefix: str) -> list[str]:
    """List all keys in R2 with a given prefix."""
    keys = []
    paginator = _get_s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def r2_delete_prefix(prefix: str):
    """Delete all objects under a given prefix."""
    keys = r2_list_keys(prefix)
    if keys:
        # S3 delete_objects accepts up to 1000 at a time
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i : i + 1000]]
            _get_s3().delete_objects(Bucket=R2_BUCKET, Delete={"Objects": batch})


# ── R2 key helpers ────────────────────────────────────────────────────────────

def _r2_image_key(image_id: str, ext: str) -> str:
    return f"images/{image_id}{ext}"


def _r2_video_key(cell_key: str) -> str:
    return f"videos/generated/{cell_key}.mp4"


def _r2_uploaded_video_key(cell_key: str) -> str:
    return f"videos/uploaded/{cell_key}.mp4"


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

    # Add r2_key column if missing (R2 migration)
    try:
        conn.execute("ALTER TABLE images ADD COLUMN r2_key TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Compute perceptual hashes for images that don't have one yet
    # For R2-backed images we skip local file hashing at startup;
    # hashes are computed at upload time.
    unhashed = conn.execute(
        "SELECT id, path, r2_key FROM images WHERE phash IS NULL"
    ).fetchall()
    if unhashed:
        hashed_count = 0
        for row in unhashed:
            local_path = row["path"]
            if local_path and Path(local_path).is_file():
                try:
                    phash = _compute_dhash(local_path)
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
    rows = conn.execute("SELECT id, filename, path, phash, r2_key FROM images").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_image(image_id: str) -> dict | None:
    conn = _get_db()
    row = conn.execute("SELECT id, filename, path, phash, r2_key FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_add_image(img_id: str, filename: str, path: str, phash: str | None = None, r2_key: str | None = None):
    conn = _get_db()
    with conn:
        conn.execute(
            "INSERT INTO images (id, filename, path, phash, r2_key) VALUES (?, ?, ?, ?, ?)",
            (img_id, filename, path, phash, r2_key),
        )
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
        r2_key = img.get("r2_key")
        exists = r2_exists(r2_key) if r2_key else Path(img["path"]).is_file()
        entry = {
            "id": img["id"],
            "filename": img["filename"],
            "url": f"/api/images/{img['id']}/file",
            "exists": exists,
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

    r2_key = img.get("r2_key")
    if r2_key:
        data = r2_get_object(r2_key)
        if data is None:
            return jsonify({"error": "File missing from R2"}), 404
        # Guess content type from extension
        ext = Path(r2_key).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(ext, "application/octet-stream")
        return Response(data, mimetype=mime)

    # Fallback: serve from local filesystem (legacy images)
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
        r2_key = _r2_image_key(img_id, ext)

        # Save to a temp file first so we can compute phash
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        phash = None
        try:
            phash = _compute_dhash(tmp_path)
        except Exception as e:
            print(f"Failed to hash uploaded image {img_id}: {e}")

        # Upload to R2
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
        content_type = mime_map.get(ext, "application/octet-stream")
        r2_upload_file(tmp_path, r2_key, content_type=content_type)

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        # Store in DB with r2_key; path is kept empty for R2-backed images
        db_add_image(img_id, f.filename, "", phash, r2_key=r2_key)
        uploaded.append({"id": img_id, "filename": f.filename, "r2_key": r2_key})

    return jsonify({"uploaded": len(uploaded), "images": uploaded})


@app.route("/api/images/<image_id>", methods=["DELETE"])
def delete_image(image_id):
    img = db_get_image(image_id)
    if not img:
        return jsonify({"error": "Not found"}), 404

    # Remove from R2
    r2_key = img.get("r2_key")
    if r2_key:
        r2_delete(r2_key)

    # Remove local file if it exists (legacy)
    if img["path"]:
        p = Path(img["path"])
        if p.is_file():
            p.unlink()

    # Remove generated videos for this image
    _remove_videos_for_image(image_id)

    db_delete_image(image_id)
    return jsonify({"message": "Deleted"})


def _remove_videos_for_image(image_id: str):
    """Delete all generated videos (on R2) that use this image."""
    prefix_gen = f"videos/generated/{image_id}__"
    prefix_up = f"videos/uploaded/{image_id}__"
    r2_delete_prefix(prefix_gen)
    r2_delete_prefix(prefix_up)


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
    """Delete all generated videos (on R2) that use this text."""
    # We need to find keys ending with __{text_id}.mp4 in both prefixes.
    # R2/S3 only supports prefix-based listing, so list all and filter.
    for prefix in ("videos/generated/", "videos/uploaded/"):
        keys = r2_list_keys(prefix)
        suffix = f"__{text_id}.mp4"
        to_delete = [k for k in keys if k.endswith(suffix)]
        if to_delete:
            for i in range(0, len(to_delete), 1000):
                batch = [{"Key": k} for k in to_delete[i : i + 1000]]
                _get_s3().delete_objects(Bucket=R2_BUCKET, Delete={"Objects": batch})


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
            gen_key = _r2_video_key(cell_key)
            uploaded_key = _r2_uploaded_video_key(cell_key)
            gs = all_status.get(cell_key)
            if gs == "generating":
                videos[cell_key] = {"status": "generating"}
            elif gs == "uploaded":
                size = r2_get_size(uploaded_key)
                if size is not None:
                    videos[cell_key] = {
                        "status": "uploaded",
                        "size": size,
                        "url": f"/api/video/{cell_key}",
                    }
            elif gs == "done":
                size = r2_get_size(gen_key)
                if size is not None:
                    videos[cell_key] = {
                        "status": "done",
                        "size": size,
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
    gen_key = _r2_video_key(safe)
    uploaded_key = _r2_uploaded_video_key(safe)

    # Try generated video first, then uploaded
    for key in (gen_key, uploaded_key):
        data = r2_get_object(key)
        if data is not None:
            return Response(data, mimetype="video/mp4")

    return jsonify({"error": "Not found"}), 404


@app.route("/api/video/<cell_key>/upload", methods=["POST"])
def mark_video_uploaded(cell_key):
    safe = Path(cell_key).name
    gen_key = _r2_video_key(safe)
    uploaded_key = _r2_uploaded_video_key(safe)

    if not r2_exists(gen_key):
        return jsonify({"error": "Video not found"}), 404

    # Copy from generated to uploaded, then delete the original
    # S3 copy_object requires the source as Bucket/Key
    _get_s3().copy_object(
        Bucket=R2_BUCKET,
        CopySource={"Bucket": R2_BUCKET, "Key": gen_key},
        Key=uploaded_key,
    )
    r2_delete(gen_key)

    db_set_gen_status(safe, "uploaded")
    return jsonify({"status": "uploaded"})


@app.route("/api/video/<cell_key>", methods=["DELETE"])
def delete_video(cell_key):
    safe = Path(cell_key).name
    r2_delete(_r2_video_key(safe))
    r2_delete(_r2_uploaded_video_key(safe))
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
        if not img or not txt:
            continue
        # Check image exists either on R2 or locally
        r2_key = img.get("r2_key")
        has_file = (r2_key and r2_exists(r2_key)) or (img["path"] and Path(img["path"]).is_file())
        if has_file:
            cell_key = f"{img['id']}__{txt['id']}"
            db_set_gen_status(cell_key, "generating")
            valid_pairs.append((img, txt, cell_key))

    if not valid_pairs:
        return jsonify({"error": "No valid pairs"}), 400

    def _run():
        for img, txt, cell_key in valid_pairs:
            # Delete old video from R2
            r2_delete(_r2_video_key(cell_key))

            VIDEO_SEMAPHORE.acquire()
            tmp_image_path = None
            tmp_video_path = None
            try:
                # Get the image to a local temp file
                r2_key = img.get("r2_key")
                if r2_key:
                    ext = Path(r2_key).suffix or ".png"
                    tmp_img_fd, tmp_image_path = tempfile.mkstemp(suffix=ext)
                    os.close(tmp_img_fd)
                    r2_download_file(r2_key, tmp_image_path)
                    actual_image_path = tmp_image_path
                else:
                    actual_image_path = img["path"]

                # Create temp file for output video
                tmp_vid_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
                os.close(tmp_vid_fd)

                render_video(
                    image_path=actual_image_path,
                    text=txt["text"],
                    output_path=tmp_video_path,
                    scroll_speed=scroll_speed,
                    hook_font_size=hook_font_size,
                    body_font_size=body_font_size,
                    audio_path=str(AUDIO_PATH),
                )

                # Upload rendered video to R2
                r2_upload_file(tmp_video_path, _r2_video_key(cell_key), content_type="video/mp4")
                db_set_gen_status(cell_key, "done")

            except Exception as e:
                db_set_gen_status(cell_key, "error")
                print(f"Error generating {cell_key}: {e}")
            finally:
                VIDEO_SEMAPHORE.release()
                # Clean up temp files
                for tmp in (tmp_image_path, tmp_video_path):
                    if tmp:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass

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
        # List all generated and uploaded videos from R2
        gen_keys = r2_list_keys("videos/generated/")
        up_keys = r2_list_keys("videos/uploaded/")
        all_keys = gen_keys + up_keys
        cells = list({Path(k).stem for k in all_keys if k.endswith(".mp4")})

    if not cells:
        return jsonify({"error": "No videos to download"}), 404

    zip_path = LIBRARY_DIR / "download.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for cell_key in cells:
            gen_key = _r2_video_key(cell_key)
            up_key = _r2_uploaded_video_key(cell_key)
            for key in (gen_key, up_key):
                video_data = r2_get_object(key)
                if video_data is not None:
                    zf.writestr(f"{cell_key}.mp4", video_data)
                    break

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
