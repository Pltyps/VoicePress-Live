import os
import sys
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from dotenv import load_dotenv
import whisper

import openai


from fastapi.middleware.cors import CORSMiddleware

# --- Load environment variables ---
base_path = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
dotenv_path = base_path / ".env"
load_dotenv(dotenv_path=dotenv_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 API key loaded:", "Yes" if openai.api_key else "Missing!")
print("👋 GPT API starting...")

# --- Init FastAPI app ---
app = FastAPI()

job_running = False 

# CORS Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pltyps.github.io"],  # The GitHub Pages domain
    # # Temporarily change for test locally
    # allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---

@app.get("/")
def home():
    return {"message": "API is working"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def get_status():
    return {"status": "processing" if job_running else "idle"}


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

# Old version
# @app.post("/upload", response_class=HTMLResponse)
# async def upload_mp4(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(".mp4"):
#         return HTMLResponse("<h3>Error: Only .mp4 files are supported.</h3>", status_code=400)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(await file.read())
#         temp_video_path = temp_video.name

#     try:
#         model = whisper.load_model("base")
#         result = model.transcribe(temp_video_path)
#         transcript = result["text"]
#         analysis = await analyze_with_transcript(transcript)

#         # HTML
#         quotes_html = "".join(f"<li>{q}</li>" for q in analysis["quotes"])
#         linkedin_html = "".join(f"<div class='card'><h4>LinkedIn Post</h4><p>{p}</p></div>" for p in analysis["social_posts"]["linkedin"])
#         instagram_html = "".join(f"<div class='card'><h4>Instagram Caption</h4><p>{p}</p></div>" for p in analysis["social_posts"]["instagram"])
#         transcript_html = f"<div class='card'><h4>Full Transcript</h4><p>{transcript}</p></div>"

#         def escape_attr(text):
#             return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

#         # Plain text (not JSON-encoded!)
#         quotes_txt = escape_attr("\n\n".join(analysis["quotes"]))
#         linkedin_txt = escape_attr("\n\n".join(analysis["social_posts"]["linkedin"]))
#         instagram_txt = escape_attr("\n\n".join(analysis["social_posts"]["instagram"]))
#         summary_txt = escape_attr(analysis["summary"])



#         html = f"""
#         <html>
#         <head>
#             <title>Interview Analysis</title>
#             <style>
#                 body {{
#                     font-family: Arial, sans-serif;
#                     margin: 2em;
#                     line-height: 1.6;
#                 }}
#                 h2 {{ color: #2c3e50; }}
#                 .card {{
#                     border: 1px solid #ddd;
#                     border-radius: 8px;
#                     padding: 1em;
#                     margin: 1em 0;
#                     background: #f9f9f9;
#                 }}
#                 ul {{ padding-left: 1.5em; }}
#             </style>
#         </head>
#         <body>
#             <button onclick="document.getElementById('videoFile').click()">⬆️ Upload Another File</button>

#             <h1>🧠 GPT Interview Summary</h1>

#             <h2>📌 Compelling Quotes</h2>
#             <button onclick="copyToClipboard(this)" data-copy-target="summary-text">📋 Copy Summary</button><pre id="summary-text">{analysis["summary"]}</pre>

#             <ul>{quotes_html}</ul>

#             <h2>📄 Summary</h2>
#             <button onclick="copyToClipboard(this)" data-copy-target="summary-text">📋 Copy Summary</button><pre id="summary-text">{analysis["summary"]}</pre>
#             <pre>{analysis["summary"]}</pre>

#             <h2>💼 LinkedIn Posts</h2>
#             <button onclick="copyToClipboard(this)" data-copy-target="summary-text">📋 Copy Summary</button><pre id="summary-text">{analysis["summary"]}</pre>
#             {linkedin_html}

#             <h2>📸 Instagram Captions</h2>
#             <button onclick="copyToClipboard(this)" data-copy-target="summary-text">📋 Copy Summary</button><pre id="summary-text">{analysis["summary"]}</pre>

#             {instagram_html}

#             <h2>📝 Full Transcript</h2>
#             {transcript_html}


#             <hr>
#             <p><a href="/upload-form">⬅️ Upload another file</a></p>

#             <script>
#                 function copyToClipboard(button) {{
#                     const targetId = button.getAttribute("data-copy-target");  // type: ignore
#                     const text = document.getElementById(targetId).innerText;
#                     navigator.clipboard.writeText(text).then(() => {{
#                         const originalText = button.innerHTML;
#                         button.innerHTML = "✅ Copied!";
#                         setTimeout(() => button.innerHTML = originalText, 1500);
#                     }}).catch(err => {{
#                         console.error("Failed to copy text:", err);
#                         button.innerHTML = "❌ Error";
#                     }});
#                 }}
#             </script>



#         </body>
#         </html>
#         """

#         return HTMLResponse(content=html)

#     except Exception as e:
#         return HTMLResponse(f"<h3>Error: {str(e)}</h3>", status_code=500)

#     finally:
#         os.remove(temp_video_path)
@app.post("/upload")
async def upload_mp4(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp4"):
        return JSONResponse({"error": "Only .mp4 files are supported."}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(temp_video_path)
        transcript = result["text"]
        analysis = await analyze_with_transcript(transcript)

        return JSONResponse({
            "summary": analysis["summary"],
            "quotes": analysis["quotes"],
            "social_posts": {
                "linkedin": analysis["social_posts"]["linkedin"],
                "instagram": analysis["social_posts"]["instagram"]
            },
            "transcript": transcript
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        os.remove(temp_video_path)

@app.post("/upload-api")
async def upload_mp4_json(file: UploadFile = File(...)):
    global job_running

    if job_running:
        return {"message": "System is currently busy. Please wait."}

    if not file.filename.lower().endswith(".mp4"):
        return {"message": "Only .mp4 files are supported."}

    job_running = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(temp_video_path)
        transcript = result["text"]
        _ = await analyze_with_transcript(transcript)

        return {"message": f"{file.filename} uploaded and processed."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

    finally:
        job_running = False
        os.remove(temp_video_path)


# --- Optional JSON POST API for direct transcript input ---
class TranscriptRequest(BaseModel):
    transcript: str

@app.post("/analyze")
async def analyze_transcript(req: TranscriptRequest):
    return await analyze_with_transcript(req.transcript)




# --- GPT Prompt Logic ---
async def analyze_with_transcript(transcript: str):
    system_message = (
        "The following is an interview transcript. You should act as a helper to do analysis and story-building for the BYU Information Systems program from this content."
        "Upon receiving an interview transcript from students, faculty, or alumni, you should organize and summarize content with attention to three primary audiences: current students, potential employers, and prospective majors."
        "Put the interviewee's name at the top of the response, return the following in valid JSON format:\n\n"
        "1. A bullet-point list of compelling quotes relevant to students, employers, and prospective majors. The quotes should be verbatim from the interview transcript.\n"
        "2. A concise 2–3 paragraph summary of the interview including introducing the interviewee, when they graduated, where they've worked, and what they've done, and any insights they have. Present information professionally, with a balance of warmth and professionalism, ensuring information is engaging and relevant to the intended audiences.\n"
        "3. 2–3 LinkedIn-style post drafts meet requirements and styles for social media post of BYU and BYU Marriott School of Business (each under 280 characters).\n"
        "4. 1–2 Instagram-style post captions meet requirements and styles for social media post of BYU and BYU Marriott School of Business (under 2200 characters, warm and friendly tone).\n\n"
        "Return this JSON:\n"
        "{\n"
        "  \"quotes\": [\"...\"],\n"
        "  \"summary\": \"...\",\n"
        "  \"social_posts\": {\n"
        "    \"linkedin\": [\"...\"],\n"
        "    \"instagram\": [\"...\"]\n"
        "  }\n"
        "}"
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
        return json.loads(reply)

    except json.JSONDecodeError as e:
        print("❌ JSON parsing error:", e)
        print("🔎 GPT raw output:", repr(reply))
        return {"error": "GPT returned invalid JSON", "raw": reply}

    except Exception as e:
        print("❌ Error calling GPT:", e)
        return {"error": str(e)}


# --- Run the app locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gpt_api_server:app", host="127.0.0.1", port=8000, reload=True)