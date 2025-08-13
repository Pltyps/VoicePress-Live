# backend/gpt_api_server.py

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

import whisper
import openai

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("voicepress-api")

def _mem_kb() -> Optional[int]:
    # Best-effort memory reporting for Linux (Render)
    try:
        import resource  # type: ignore
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

def log_memory(tag: str):
    kb = _mem_kb()
    if kb is not None:
        logger.info(f"🧠 {tag} | ru_maxrss={kb} KB")

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

# ---------- Whisper preload ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")  # tiny/base/small…
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")  # set to 'en' for speed if applicable

logger.info(f"🧠 Preloading Whisper model ({WHISPER_MODEL})...")
model = whisper.load_model(WHISPER_MODEL)
logger.info("✅ Whisper model loaded and ready")
log_memory("after_model_load")

# ---------- FastAPI ----------
app = FastAPI()
job_running = False

# CORS for GitHub Pages (your site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pltyps.github.io",
    ],
    allow_origin_regex=r"https://.*\.github\.io$",   # covers org/user pages if you switch
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

# ---------- Routes ----------
@app.head("/")
def head_root():
    # Render sometimes probes with HEAD; avoid 405
    return Response(status_code=200)

@app.get("/")
def home():
    logger.info("✅ GET / - API health check")
    return {"message": "API is working"}

@app.get("/health")
def health():
    logger.info("✅ GET /health - Health check OK")
    return {"status": "ok"}

@app.get("/status")
def get_status():
    status = "processing" if job_running else "idle"
    logger.info(f"📡 GET /status - System is {status}")
    return {"status": status}

@app.post("/log-error")
async def log_error(err: FrontendError):
    # Optional sink for frontend to report issues (e.g., 502/503)
    logger.warning(f"🟠 Frontend error: {err.type} | {err.statusCode} | {err.message}")
    if err.extra:
        logger.warning(f"🟠 Extra: {json.dumps(err.extra)[:500]}")
    return {"ok": True}

@app.get("/upload-form", response_class=HTMLResponse)
def upload_form():
    logger.info("🧾 GET /upload-form - Serving HTML upload form")
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
    logger.info(f"📥 POST /upload - Received file: {file.filename}")

    if not file.filename.lower().endswith(".mp4"):
        logger.warning("❌ Rejected: File is not .mp4")
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    temp_path = None
    try:
        job_running = True

        # Stream to temp file (avoids big in-memory read)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                temp_video.write(chunk)

        logger.info(f"📁 File saved to temp path: {temp_path}")
        log_memory("after_save")

        # Transcribe
        logger.info(f"🎙️ Transcribing audio with Whisper ({WHISPER_MODEL})...")
        result = model.transcribe(
            temp_path,
            language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
            fp16=False  # CPU on Render
        )
        transcript = result.get("text", "").strip()
        logger.info("📝 Transcription complete")

        # Analyze with GPT
        logger.info("🤖 Sending transcript to GPT...")
        analysis = await analyze_with_transcript(transcript)

        if "error" in analysis:
            return JSONResponse({"error": analysis["error"]}, status_code=500)

        logger.info("✅ Upload and analysis successful")
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
        if temp_path:
            try:
                os.remove(temp_path)
                logger.info(f"🧹 Temp file deleted: {temp_path}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

@app.post("/upload-api")
async def upload_mp4_json(file: UploadFile = File(...)):
    """
    JSON-only variant; reuses the same preloaded Whisper model.
    """
    global job_running
    logger.info(f"📥 POST /upload-api - Received: {file.filename}")

    if job_running:
        logger.warning("🚫 System busy, rejecting new upload")
        return JSONResponse({"message": "System is currently busy. Please wait."}, status_code=429)

    if not file.filename.lower().endswith(".mp4"):
        logger.warning("❌ Rejected non-MP4 file")
        return JSONResponse({"message": "Only .mp4 files are supported."}, status_code=400)

    temp_path = None
    try:
        job_running = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_video.write(chunk)

        logger.info("📥 File stored temporarily")
        log_memory("after_save_json")

        result = model.transcribe(temp_path, language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None, fp16=False)
        transcript = result.get("text", "").strip()

        _ = await analyze_with_transcript(transcript)

        logger.info("✅ JSON upload and analysis complete")
        return JSONResponse({"message": f"{file.filename} uploaded and processed."})

    except Exception as e:
        logger.exception("💥 Error during /upload-api processing")
        return JSONResponse({"message": f"Error: {str(e)}"}, status_code=500)

    finally:
        job_running = False
        if temp_path:
            try:
                os.remove(temp_path)
                logger.info(f"🧹 Cleaned up temp file: {temp_path}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

@app.post("/analyze")
async def analyze_transcript(req: TranscriptRequest):
    logger.info("📤 POST /analyze - Manual transcript submitted")
    return await analyze_with_transcript(req.transcript)

# ---------- GPT helper ----------
async def analyze_with_transcript(transcript: str):
    logger.info("🧠 Calling GPT with transcript content...")
    system_message = (
        "You are a content assistant. Given an interview transcript, produce JSON with keys: "
        "`summary` (<= 8 sentences), `quotes` (array of punchy quotes), and `social_posts` with "
        "subkeys `linkedin` (2 short posts) and `instagram` (2 captions). Respond ONLY with JSON."
    )

    try:
        # openai==0.28.0 (as in requirements) uses ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcript or "(empty transcript)"}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message["content"]
        logger.info("✅ GPT returned a response")
        return json.loads(reply)

    except json.JSONDecodeError:
        logger.exception("💥 GPT returned malformed JSON")
        return {"error": "GPT returned invalid JSON. Check system prompt or model output parsing."}

    except Exception as e:
        logger.exception("💥 Error calling OpenAI GPT")
        return {"error": str(e)}

# ---------- Local entry ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
