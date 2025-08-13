import os
import sys
import json
import time
import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import whisper
import openai
from fastapi.middleware.cors import CORSMiddleware

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voicepress-api")

# --- Load environment variables ---
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

# --- Memory helper (no extra deps) ---
def log_mem(tag: str):
    try:
        import resource  # available on Linux
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(f"🧠 {tag} | ru_maxrss={rss_kb} KB")
    except Exception:
        pass

# Load Whisper model globally at startup
logger.info("🧠 Preloading Whisper model (tiny)...")
model = whisper.load_model("tiny")
logger.info("✅ Whisper model loaded and ready")
log_mem("after_model_load")

# --- Init FastAPI app ---
app = FastAPI()
job_running = False

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pltyps.github.io"],  # your GitHub Pages origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_marker(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-App"] = "voicepress-live"
    return response

# --- Routes ---
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

# Optional explicit preflight (CORSMiddleware would handle this anyway)
@app.options("/{full_path:path}")
async def preflight_handler():
    return Response(status_code=200)

@app.options("/upload")
async def upload_options():
    logger.info("⚙️ OPTIONS /upload - Preflight request")
    return Response(status_code=200)

@app.post("/upload")
async def upload_mp4(file: UploadFile = File(...)):
    global job_running
    logger.info(f"📥 POST /upload - Received file: {file.filename}")

    if not file.filename.lower().endswith(".mp4"):
        logger.warning("❌ Rejected: File is not .mp4")
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    temp_path = None
    start_total = time.time()
    try:
        # Stream to disk to avoid loading entire file into RAM
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_path = temp_video.name
            # 1MB chunks
            shutil.copyfileobj(file.file, temp_video, length=1024 * 1024)

        logger.info(f"📁 File saved to temp path: {temp_path}")
        log_mem("after_save")

        # Mark processing so /status shows correctly
        job_running = True

        # Transcribe
        t0 = time.time()
        logger.info("🎙️ Transcribing audio with Whisper (tiny)...")
        result = model.transcribe(temp_path)
        transcript = result.get("text", "")
        logger.info(f"📝 Transcription complete in {time.time() - t0:.2f}s")
        log_mem("after_transcribe")

        # Analyze with GPT
        t1 = time.time()
        logger.info("🤖 Sending transcript to GPT...")
        analysis = await analyze_with_transcript(transcript)
        logger.info(f"✅ GPT analysis complete in {time.time() - t1:.2f}s")
        log_mem("after_gpt")

        logger.info(f"✅ Upload+analysis successful in {time.time() - start_total:.2f}s")
        return JSONResponse({
            "summary": analysis.get("summary"),
            "quotes": analysis.get("quotes"),
            "social_posts": {
                "linkedin": analysis.get("social_posts", {}).get("linkedin", []),
                "instagram": analysis.get("social_posts", {}).get("instagram", []),
            },
            "transcript": transcript,
        })

    except Exception as e:
        logger.exception("💥 Error during /upload processing")
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        job_running = False
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"🧹 Temp file deleted: {temp_path}")
        except Exception as cleanup_err:
            logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

@app.post("/upload-api")
async def upload_mp4_json(file: UploadFile = File(...)):
    """Alt endpoint (JSON-only message). Reuses the global Whisper model."""
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
            shutil.copyfileobj(file.file, temp_video, length=1024 * 1024)

        logger.info("📥 File stored temporarily (streamed)")
        result = model.transcribe(temp_path)
        transcript = result.get("text", "")
        _ = await analyze_with_transcript(transcript)

        logger.info("✅ JSON upload and analysis complete")
        return JSONResponse({"message": f"{file.filename} uploaded and processed."})

    except Exception as e:
        logger.exception("💥 Error during /upload-api processing")
        return JSONResponse({"message": f"Error: {str(e)}"}, status_code=500)

    finally:
        job_running = False
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"🧹 Cleaned up temp file: {temp_path}")
        except Exception as cleanup_err:
            logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

class TranscriptRequest(BaseModel):
    transcript: str

@app.post("/analyze")
async def analyze_transcript(req: TranscriptRequest):
    logger.info("📤 POST /analyze - Manual transcript submitted")
    return await analyze_with_transcript(req.transcript)

async def analyze_with_transcript(transcript: str):
    logger.info("🧠 Calling GPT with transcript content...")
    system_message = (
        "You are an assistant that returns strictly JSON with keys "
        '["summary","quotes","social_posts"]. '
        '"quotes" is a list of short quotes. '
        '"social_posts" contains {"linkedin": [...], "instagram": [...]}. '
        "No extra text outside JSON."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcript},
            ],
            temperature=0.2,
        )
        reply = response.choices[0].message.content
        logger.info("✅ GPT returned a response")
        return json.loads(reply)

    except json.JSONDecodeError:
        logger.exception("💥 GPT returned malformed JSON")
        return {"error": "GPT returned invalid JSON", "raw": reply}

    except Exception as e:
        logger.exception("💥 Error calling OpenAI GPT")
        return {"error": str(e)}

# --- Local development entry point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
