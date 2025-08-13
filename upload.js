const BACKEND_URL = "https://voicepress-live-api.onrender.com";
// const BACKEND_URL = "http://127.0.0.1:8000"; // for local testing

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("videoFile");
  const statusEl = document.getElementById("uploadStatus");
  const spinner = document.getElementById("spinner");
  const progressBar = document.getElementById("uploadProgress");
  const output = document.getElementById("output");
  const fileNameDisplay = document.getElementById("fileName");
  const submitButton = form.querySelector("button");
  const statusLight = document.getElementById("statusLight");

  // ---- helpers ----
  let processingPoller = null;
  let normalPoller = null;

  function setStage(message, opts = {}) {
    statusEl.textContent = message;
    if (opts.showSpinner !== undefined)
      spinner.style.display = opts.showSpinner ? "block" : "none";
    if (opts.progress === "hide") {
      progressBar.style.display = "none";
    } else if (opts.progress === "determinate") {
      progressBar.style.display = "block";
      // determinate state -> must have value
      if (!progressBar.hasAttribute("value")) progressBar.value = 0;
    } else if (opts.progress === "indeterminate") {
      progressBar.style.display = "block";
      // indeterminate state -> remove value attr
      progressBar.removeAttribute("value");
    }
    if (opts.disableButton !== undefined)
      submitButton.disabled = opts.disableButton;
  }

  // speed up status polling while a job is running
  async function pollStatusOnce() {
    try {
      const res = await fetch(`${BACKEND_URL}/status`, { cache: "no-store" });
      const data = await res.json();
      if (data.status === "processing") {
        statusLight.textContent =
          "🔴 Processing… Generating transcript, summary, quotes, and social posts.";
        statusLight.style.color = "#e74c3c";
        submitButton.disabled = true;
      } else {
        statusLight.textContent =
          "🟢 System is ready. You can upload a video file.";
        statusLight.style.color = "#2ecc71";
        submitButton.disabled = false;
      }
    } catch {
      statusLight.textContent =
        "⚠️ Unable to retrieve system status. Please refresh.";
      statusLight.style.color = "#f39c12";
    }
  }

  function startNormalPolling() {
    stopProcessingPolling();
    if (normalPoller) clearInterval(normalPoller);
    normalPoller = setInterval(pollStatusOnce, 5000);
    pollStatusOnce();
  }

  function startProcessingPolling() {
    if (normalPoller) clearInterval(normalPoller);
    if (processingPoller) clearInterval(processingPoller);
    processingPoller = setInterval(pollStatusOnce, 1500);
    pollStatusOnce();
  }

  function stopProcessingPolling() {
    if (processingPoller) {
      clearInterval(processingPoller);
      processingPoller = null;
    }
  }

  // show selected file name
  fileInput.addEventListener("change", () => {
    fileNameDisplay.textContent = fileInput.files.length
      ? fileInput.files[0].name
      : "No file chosen";
  });

  // ---- upload handler ----
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      console.warn("🚫 No file selected.");
      return;
    }

    // reset UI
    output.innerHTML = "";
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BACKEND_URL}/upload`, true);
    xhr.timeout = 1000 * 60 * 30; // 30 minutes (adjust as needed)

    xhr.onloadstart = () => {
      setStage("Uploading…", {
        showSpinner: true,
        progress: "determinate",
        disableButton: true,
      });
      progressBar.value = 0;
      startProcessingPolling(); // begin tighter polling immediately
      console.log("📤 Upload started:", file.name);
    };

    // upload progress (deterministic)
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        progressBar.value = percent;
        statusEl.textContent = `Uploading… ${percent.toFixed(1)}%`;
      }
    });

    // when the upload has finished sending bytes, but server is still working
    xhr.upload.addEventListener("load", () => {
      setStage(
        "Upload complete. Processing (extracting audio → transcribing → summarizing)…",
        {
          showSpinner: true,
          progress: "indeterminate",
          disableButton: true,
        }
      );
      console.log("📦 Upload finished; server is processing…");
    });

    // network-level errors
    xhr.onerror = () => {
      stopProcessingPolling();
      setStage("❌ Upload failed due to a network error.", {
        showSpinner: false,
        progress: "hide",
        disableButton: false,
      });
      output.innerHTML = `<p style="color:red;">Network error (XHR status ${
        xhr.status || 0
      }).</p>`;
    };

    xhr.ontimeout = () => {
      stopProcessingPolling();
      setStage(
        "⏰ Request timed out while processing. Please try again later.",
        {
          showSpinner: false,
          progress: "hide",
          disableButton: false,
        }
      );
      output.innerHTML = `<p style="color:red;">The server took too long to respond.</p>`;
    };

    // final response
    xhr.onload = () => {
      stopProcessingPolling();
      submitButton.disabled = false;

      // restore progress UI
      spinner.style.display = "none";
      progressBar.style.display = "none";

      let response = {};
      try {
        response = JSON.parse(xhr.responseText || "{}");
      } catch {
        setStage("❌ Server returned invalid JSON.", {
          showSpinner: false,
          progress: "hide",
        });
        output.innerHTML = `<p style="color:red;">Invalid server response</p>`;
        fileInput.value = "";
        fileNameDisplay.textContent = "No file chosen";
        startNormalPolling();
        return;
      }

      // friendly errors
      if (xhr.status === 200) {
        setStage("✅ Processing complete!", {
          showSpinner: false,
          progress: "hide",
        });
        renderContent(response);
      } else {
        const friendly =
          xhr.status === 429
            ? "🚦 System is busy. Please wait for the current job to finish."
            : xhr.status === 413
            ? "📦 File too large for the server limits."
            : xhr.status === 502 || xhr.status === 503 || xhr.status === 504
            ? "🚨 The server restarted or ran out of memory during processing."
            : response.error || "❌ Upload/processing failed.";

        setStage(friendly, { showSpinner: false, progress: "hide" });
        output.innerHTML = `<p style="color:red;">${friendly}</p>`;
      }

      // cleanup selection
      fileInput.value = "";
      fileNameDisplay.textContent = "No file chosen";
      startNormalPolling();
    };

    xhr.send(formData);
  });

  // ---- background status light ----
  function initStatusLight() {
    // start the normal slower poll; we’ll swap to faster during processing
    if (normalPoller) clearInterval(normalPoller);
    normalPoller = setInterval(pollStatusOnce, 5000);
    pollStatusOnce();
  }

  initStatusLight();
});

// ---- render GPT response content (unchanged) ----
function renderContent(data) {
  const output = document.getElementById("output");

  function createSection(title, content, copyText) {
    return `
      <div class="card">
        <h2>${title}</h2>
        <pre>${escapeHtml(content)}</pre>
        <button class="copy-btn" onclick="copyToClipboard(\`${escapeHtml(
          copyText
        )}\`)">📋 Copy</button>
        <div style="clear: both;"></div>
      </div>
    `;
  }

  const quotes = (data.quotes || []).join("\n\n");
  const linkedin = (data.social_posts?.linkedin || []).join("\n\n");
  const instagram = (data.social_posts?.instagram || []).join("\n\n");

  output.innerHTML = `
    <h1>🧠 GPT Interview Summary</h1>
    ${createSection("📌 Compelling Quotes", quotes, quotes)}
    ${createSection("📄 Summary", data.summary || "", data.summary || "")}
    ${createSection("💼 LinkedIn Posts", linkedin, linkedin)}
    ${createSection("📸 Instagram Captions", instagram, instagram)}
    ${createSection(
      "📝 Full Transcript",
      data.transcript || "",
      data.transcript || ""
    )}
  `;
}

// ---- utils ----
function escapeHtml(str = "") {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      Toastify({
        text: "✅ Copied to clipboard!",
        duration: 2000,
        gravity: "top",
        position: "right",
        backgroundColor: "#4CAF50",
      }).showToast();
    })
    .catch(() => {
      Toastify({
        text: "❌ Failed to copy.",
        duration: 2000,
        gravity: "top",
        position: "right",
        backgroundColor: "#FF4C4C",
      }).showToast();
    });
}
