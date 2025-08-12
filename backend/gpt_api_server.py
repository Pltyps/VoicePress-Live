import os
import sys
import json
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
    base_path = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    dotenv_path = base_path / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing from environment")

    logger.info("🔑 OPENAI_API_KEY loaded successfully")
except Exception as e:
    logger.critical("💥 Failed to load .env or API key", exc_info=True)
    sys.exit(1)

logger.info("👋 GPT API server starting...")

# --- Init FastAPI app ---
app = FastAPI()
job_running = False

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pltyps.github.io"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    allow_credentials=True,
    expose_headers=["Content-Type"],
    max_age=86400,
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

@app.options("/upload")
async def upload_options():
    logger.info("⚙️ OPTIONS /upload - Preflight request")
    return Response(status_code=200)

@app.post("/upload")
async def upload_mp4(file: UploadFile = File(...)):
    logger.info(f"📥 POST /upload - Received file: {file.filename}")

    if not file.filename.lower().endswith(".mp4"):
        logger.warning("❌ Rejected: File is not .mp4")
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_path = temp_video.name

        logger.info(f"📁 File saved to temp path: {temp_path}")

        logger.info("🧠 Loading Whisper model...")
        model = whisper.load_model("base")

        logger.info("🎙️ Transcribing audio...")
        result = model.transcribe(temp_path)
        transcript = result["text"]
        logger.info("📝 Transcription complete")

        logger.info("🤖 Sending transcript to GPT...")
        analysis = await analyze_with_transcript(transcript)

        logger.info("✅ Upload and analysis successful")
        return JSONResponse({
            "summary": analysis.get("summary"),
            "quotes": analysis.get("quotes"),
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
        try:
            os.remove(temp_path)
            logger.info(f"🧹 Temp file deleted: {temp_path}")
        except Exception as cleanup_err:
            logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")

@app.post("/upload-api")
async def upload_mp4_json(file: UploadFile = File(...)):
    global job_running
    logger.info(f"📥 POST /upload-api - Received: {file.filename}")

    if job_running:
        logger.warning("🚫 System busy, rejecting new upload")
        return JSONResponse({"message": "System is currently busy. Please wait."}, status_code=429)

    if not file.filename.lower().endswith(".mp4"):
        logger.warning("❌ Rejected non-MP4 file")
        return JSONResponse({"message": "Only .mp4 files are supported."}, status_code=400)

    job_running = True
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_path = temp_video.name

        logger.info("📥 File stored temporarily")
        model = whisper.load_model("base")
        result = model.transcribe(temp_path)
        transcript = result["text"]

        _ = await analyze_with_transcript(transcript)

        logger.info("✅ JSON upload and analysis complete")
        return JSONResponse({"message": f"{file.filename} uploaded and processed."})

    except Exception as e:
        logger.exception("💥 Error during /upload-api processing")
        return JSONResponse({"message": f"Error: {str(e)}"}, status_code=500)

    finally:
        job_running = False
        try:
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
        "The following is an interview transcript... [shortened here for clarity; full version kept in code]"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcript}
            ]
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
    logger.info("🚀 Starting Uvicorn dev server")
    uvicorn.run("gpt_api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
