import os
import sys
import json
import tempfile
import logging
import uuid
import shutil
import subprocess
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

import openai

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
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
        logger.info(
            f"💽 /tmp ({tag}) -> total={total/1_073_741_824:.2f} GiB, "
            f"used={used/1_073_741_824:.2f} GiB, free={free/1_073_741_824:.2f} GiB"
        )
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

# ---------- FastAPI ----------
app = FastAPI()
job_running = False

# CORS for GitHub Pages (your site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pltyps.github.io",
    ],
    allow_origin_regex=r"https://.*\.github\.io$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# ---------- Helpers: streaming MP4 -> audio (server keeps RAM tiny) ----------
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "flac")   # "flac" (lossless, smaller than WAV) or "aac"
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "96k")  # only used if aac
AUDIO_RATE = os.getenv("AUDIO_RATE", "16000")    # 16 kHz for Whisper
AUDIO_MONO = "1"

def _ffmpeg_cmd(out_path: str):
    # ffmpeg will read MP4 from stdin (pipe:0) and write audio only to out_path
    if AUDIO_CODEC.lower() == "aac":
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-vn", "-ac", AUDIO_MONO, "-ar", AUDIO_RATE,
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            out_path,
        ]
    # default: FLAC (lossless)
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-vn", "-ac", AUDIO_MONO, "-ar", AUDIO_RATE,
        "-c:a", "flac",
        out_path,
    ]

async def stream_mp4_to_audio(upload: UploadFile) -> str:
    """
    Streams the incoming MP4 bytes directly into ffmpeg via stdin.
    Only an audio file is written to /tmp. Memory stays low.
    Returns the audio temp file path.
    """
    suffix = ".m4a" if AUDIO_CODEC.lower() == "aac" else ".flac"
    out_path = f"/tmp/audio-{uuid.uuid4().hex}{suffix}"

    proc = subprocess.Popen(_ffmpeg_cmd(out_path), stdin=subprocess.PIPE)
    try:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            proc.stdin.write(chunk)
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"ffmpeg failed with code {ret}")
        return out_path
    except Exception:
        proc.kill()
        raise

# ---------- Transcription with OpenAI Whisper API ----------
WHISPER_API_MODEL = os.getenv("WHISPER_API_MODEL", "whisper-1")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")  # optional hint

def transcribe_with_openai(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            try:
                resp = openai.Audio.transcriptions.create(
                    model=WHISPER_API_MODEL,
                    file=f,
                    language=WHISPER_LANGUAGE
                )
                text = getattr(resp, "text", None) or resp.get("text", "")
                return text.strip()
            except Exception:
                f.seek(0)
                resp = openai.Audio.transcribe(
                    WHISPER_API_MODEL, f, language=WHISPER_LANGUAGE
                )
                text = resp.get("text", "")
                return text.strip()
    except Exception as e:
        logger.exception("💥 OpenAI Whisper transcription failed")
        raise RuntimeError(str(e))

# ---------- Verbatim quotes helpers ----------
def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _filter_verbatim_quotes(transcript: str, quotes: list[str], max_quotes: int = 8) -> list[str]:
    t_norm = _normalize_spaces(transcript).lower()
    kept = []
    seen = set()
    for q in quotes or []:
        q_clean = (q or "").strip()
        if not q_clean:
            continue
        q_norm = _normalize_spaces(q_clean).lower()
        if q_norm and q_norm in t_norm:
            if q_norm not in seen:
                kept.append(q_clean)
                seen.add(q_norm)
        if len(kept) >= max_quotes:
            break
    return kept

# ---------- Routes ----------
@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/")
def home():
    return {"message": "API is working"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def get_status():
    status = "processing" if job_running else "idle"
    return {"status": status}

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
async def upload_mp4(file: UploadFile = File(...)):
    global job_running
    if not file.filename.lower().endswith(".mp4"):
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    audio_path = None
    try:
        if job_running:
            return JSONResponse({"error": "System is currently busy. Please wait."}, status_code=429)
        job_running = True

        audio_path = await stream_mp4_to_audio(file)
        transcript = transcribe_with_openai(audio_path)
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
        job_running = False
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
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
        return {"error": "GPT returned invalid JSON."}
    except Exception as e:
        return {"error": str(e)}

# ---------- Local entry ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
