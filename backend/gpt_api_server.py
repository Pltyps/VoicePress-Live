import os
import sys
import json
import logging
import uuid
import shutil
import subprocess
import re
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum

import openai

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voicepress-api")

def _mem_kb() -> Optional[int]:
    try:
        import resource  # type: ignore
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

def log_memory(tag: str):
    kb = _mem_kb()
    if kb is not None:
        logger.info(f"🧠 {tag} | ru_maxrss={kb} KB")

def log_tmp_disk(tag: str = "boot"):
    try:
        total, used, free = shutil.disk_usage("/tmp")
        logger.info(f"💽 /tmp ({tag}) -> total={total/1_073_741_824:.2f} GiB, used={used/1_073_741_824:.2f} GiB, free={free/1_073_741_824:.2f} GiB")
    except Exception:
        logger.info("💽 /tmp capacity check failed")

# ---------- Env / OpenAI ----------
try:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    dotenv_path = base_path / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing from environment")

    logger.info("🔑 OPENAI_API_KEY loaded successfully")
except Exception:
    logger.critical("💥 Failed to load .env or API key", exc_info=True)
    sys.exit(1)

logger.info("👋 GPT API server starting...")
log_tmp_disk("startup")

# ---------- FastAPI / State ----------
app = FastAPI()

class Stage(str, Enum):
    idle = "idle"
    extracting = "extracting"
    transcribing = "transcribing"
    summarizing = "summarizing"

job_stage: Stage = Stage.idle
job_running = False
_job_lock = threading.Lock()

def set_stage(s: Stage):
    global job_stage
    job_stage = s

# CORS for GitHub Pages (your site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pltyps.github.io"],
    allow_origin_regex=r"https://.*\.github\.io$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-App"],
    max_age=86400,
)

@app.middleware("http")
async def add_marker(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-App"] = "voicepress-live"
    return resp

# ---------- Models ----------
class TranscriptRequest(BaseModel):
    transcript: str

class FrontendError(BaseModel):
    type: str
    message: str
    statusCode: Optional[int] = None
    time: Optional[str] = None
    extra: Optional[dict] = None

# ---------- Config for audio extraction ----------
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "flac")   # "flac" or "aac"
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "96k")  # only for aac
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")    # 16 kHz
AUDIO_MONO = "1"
FFMPEG_TIMEOUT_SECS = int(os.getenv("FFMPEG_TIMEOUT_SECS", "3600"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(3 * 1024 * 1024 * 1024)))  # 3 GB default

# ---------- Robust video save + audio extract ----------
def _ffmpeg_extract_audio(src_path: str, out_path: str):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src_path, "-vn", "-ac", AUDIO_MONO, "-ar", AUDIO_RATE]
    if AUDIO_CODEC.lower() == "aac":
        cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, out_path]
    else:
        cmd += ["-c:a", "flac", out_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=FFMPEG_TIMEOUT_SECS)
    if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:500]}")

async def _save_upload_to_temp_mp4(upload: UploadFile) -> str:
    tmp_video = f"/tmp/in-{uuid.uuid4().hex}.mp4"
    with open(tmp_video, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
    return tmp_video

async def extract_audio_from_upload(upload: UploadFile) -> str:
    """Save full mp4 to disk, then extract audio. Robust for MP4s with moov at end."""
    src_path = await _save_upload_to_temp_mp4(upload)
    suffix = ".m4a" if AUDIO_CODEC.lower() == "aac" else ".flac"
    out_path = f"/tmp/audio-{uuid.uuid4().hex}{suffix}"
    try:
        set_stage(Stage.extracting)
        _ffmpeg_extract_audio(src_path, out_path)
        return out_path
    finally:
        try:
            os.remove(src_path)
        except Exception:
            pass

# ---------- Whisper API (openai==0.28.0 style) ----------
WHISPER_API_MODEL = os.getenv("WHISPER_API_MODEL", "whisper-1")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")  # optional hint

def _get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def transcribe_with_openai(audio_path: str) -> str:
    # avoid API's 0.1s minimum error on truly empty/failed extractions
    duration = _get_audio_duration(audio_path)
    if duration < 0.1:
        logger.warning(f"⏳ Extracted audio too short ({duration:.3f}s). Skipping transcription.")
        return ""

    set_stage(Stage.transcribing)
    try:
        with open(audio_path, "rb") as f:
            # Correct method for SDK 0.28.0:
            resp = openai.Audio.transcribe(WHISPER_API_MODEL, f, language=WHISPER_LANGUAGE)
            return (resp.get("text") or "").strip()
    except Exception as e:
        logger.exception("💥 OpenAI Whisper transcription failed")
        raise RuntimeError(str(e))

# ---------- Verbatim quotes helpers ----------
def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _filter_verbatim_quotes(transcript: str, quotes: list[str], max_quotes: int = 8) -> list[str]:
    t_norm = _normalize_spaces(transcript).lower()
    kept, seen = [], set()
    for q in quotes or []:
        q_clean = (q or "").strip()
        if not q_clean:
            continue
        q_norm = _normalize_spaces(q_clean).lower()
        if q_norm and q_norm in t_norm and q_norm not in seen:
            kept.append(q_clean)
            seen.add(q_norm)
        if len(kept) >= max_quotes:
            break
    return kept

# ---------- Routes ----------
@app.head("/")
def head_root():
    return Response(status_code=200)

# Catch-all OPTIONS so preflight always succeeds when app is up
@app.options("/{rest_of_path:path}")
def options_cors(rest_of_path: str):
    return Response(status_code=204)

@app.get("/")
def home():
    return {"message": "API is working"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def get_status():
    # Backward compatibility: return "processing" like your old UI expects,
    # and also include the richer stage.
    status = "processing" if job_stage != Stage.idle else "idle"
    return {"status": status, "stage": job_stage}

@app.get("/debug/tmp")
def debug_tmp():
    total, used, free = shutil.disk_usage("/tmp")
    return {
        "tmp_total_gib": round(total/1_073_741_824, 3),
        "tmp_used_gib": round(used/1_073_741_824, 3),
        "tmp_free_gib": round(free/1_073_741_824, 3),
    }

@app.post("/log-error")
async def log_error(err: FrontendError):
    return {"ok": True}

@app.get("/upload-form", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
      <head><title>Upload MP4 Interview</title></head>
      <body>
        <h2>Upload an MP4 Interview File</h2>
        <form action="/upload" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept=".mp4" required>
          <input type="submit" value="Upload and Analyze">
        </form>
      </body>
    </html>
    """

@app.post("/upload")
async def upload_mp4(file: UploadFile = File(...), request: Request = None):
    global job_running
    if not file.filename.lower().endswith(".mp4"):
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    # optional fast fail based on Content-Length
    try:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "File too large."}, status_code=413)
    except Exception:
        pass

    audio_path = None
    with _job_lock:
        if job_running:
            return JSONResponse({"error": "System is currently busy. Please wait."}, status_code=429)
        job_running = True
        set_stage(Stage.extracting)

    try:
        # 1) Save full video & extract audio (robust)
        audio_path = await extract_audio_from_upload(file)
        log_tmp_disk("after_extract")

        # 2) Transcribe with OpenAI Whisper API (0.28.0 method)
        transcript = transcribe_with_openai(audio_path)
        log_memory("after_transcribe")

        # 3) Analyze with GPT
        set_stage(Stage.summarizing)
        analysis = await analyze_with_transcript(transcript)
        if "error" in analysis:
            return JSONResponse({"error": analysis["error"]}, status_code=500)

        return JSONResponse({
            "summary": analysis.get("summary", ""),
            "quotes": analysis.get("quotes", []),
            "social_posts": {
                "linkedin": analysis.get("social_posts", {}).get("linkedin", []),
                "instagram": analysis.get("social_posts", {}).get("instagram", [])
            },
            "transcript": transcript
        })

    except Exception as e:
        logger.exception("💥 Error during /upload processing")
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        set_stage(Stage.idle)
        with _job_lock:
            job_running = False
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"🧹 Temp audio deleted: {audio_path}")
            except Exception:
                pass

# ---------- GPT helper ----------
async def analyze_with_transcript(transcript: str):
    system_message = (
        "You are a careful content assistant. Given a human interview transcript, return STRICT JSON with keys:\n"
        "  - summary: <= 8 sentences\n"
        "  - quotes: array of compelling DIRECT QUOTES that appear VERBATIM in the transcript\n"
        "  - social_posts: { linkedin: [2 short posts], instagram: [2 captions] }\n\n"
        "Rules:\n"
        "1) All items in 'quotes' MUST be EXACT substrings from the transcript (verbatim). Do not paraphrase.\n"
        "2) If you cannot find verbatim quotes, return an empty quotes array.\n"
        "3) Respond ONLY with JSON. No commentary.\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcript or "(empty transcript)"}
            ],
            temperature=0.2
        )
        reply = response.choices[0].message["content"]
        data = json.loads(reply)

        if isinstance(data, dict):
            raw_quotes = data.get("quotes", [])
            data["quotes"] = _filter_verbatim_quotes(transcript, raw_quotes, max_quotes=8)
            data.setdefault("summary", "")
            sp = data.get("social_posts") or {}
            sp.setdefault("linkedin", [])
            sp.setdefault("instagram", [])
            data["social_posts"] = sp
        return data

    except json.JSONDecodeError:
        logger.exception("💥 GPT returned malformed JSON")
        return {"error": "GPT returned invalid JSON. Check system prompt or model output parsing."}
    except Exception as e:
        logger.exception("💥 Error calling OpenAI GPT")
        return {"error": str(e)}

# ---------- Local entry ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render runs uvicorn on 10000 in your logs
    logger.info(f"🚀 Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
