const BACKEND_URL = "https://voicepress-live-api.onrender.com";
// const BACKEND_URL = "http://127.0.0.1:8000"; // for local testing

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("videoFile");
  const status = document.getElementById("uploadStatus");
  const spinner = document.getElementById("spinner");
  const progressBar = document.getElementById("uploadProgress");
  const output = document.getElementById("output");
  const fileNameDisplay = document.getElementById("fileName");
  const submitButton = form.querySelector("button");

  // Display selected file name
  fileInput.addEventListener("change", () => {
    fileNameDisplay.textContent = fileInput.files.length
      ? fileInput.files[0].name
      : "No file chosen";
  });

  // Upload handler
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      console.warn("🚫 No file selected.");
      return;
    }

    console.log("📤 Upload started:", file.name);
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BACKEND_URL}/upload`, true);

    xhr.onloadstart = () => {
      spinner.style.display = "block";
      progressBar.style.display = "block";
      progressBar.value = 0;
      status.textContent = "Uploading...";
    };

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        console.log(`📶 Upload progress: ${percent.toFixed(1)}%`);
        progressBar.value = percent;
      }
    });

    xhr.onload = () => {
      spinner.style.display = "none";
      progressBar.style.display = "none";
      submitButton.disabled = false;

      try {
        const response = JSON.parse(xhr.responseText);

        if (xhr.status === 200) {
          console.log("✅ Upload successful:", response);
          status.textContent = "✅ Upload complete!";
          renderContent(response);
        } else {
          console.error("❌ Upload failed with status", xhr.status, response);
          status.textContent = "❌ Upload failed.";
          output.innerHTML = `<p style="color:red;">${response.error}</p>`;
        }
      } catch (err) {
        console.error("❌ Failed to parse JSON from server:", err);
        status.textContent = "❌ Upload failed.";
        output.innerHTML = `<p style="color:red;">Invalid server response</p>`;
      }

      fileInput.value = "";
      fileNameDisplay.textContent = "No file chosen";
    };

    xhr.onerror = () => {
      spinner.style.display = "none";
      progressBar.style.display = "none";
      submitButton.disabled = false;
      console.error("❌ Upload failed due to a network error.");
      status.textContent = "❌ Upload failed.";
    };

    xhr.send(formData);
  });

  // Status polling
  async function pollStatus() {
    const light = document.getElementById("statusLight");

    try {
      const res = await fetch(`${BACKEND_URL}/status`);
      const data = await res.json();
      console.log("📡 Polled status:", data.status);

      if (data.status === "processing") {
        light.textContent =
          "🔴 Processing... Generating transcript, summary, quotes, and social posts.";
        light.style.color = "#e74c3c";
        submitButton.disabled = true;
      } else {
        light.textContent = "🟢 System is ready. You can upload a video file.";
        light.style.color = "#2ecc71";
        submitButton.disabled = false;
      }
    } catch (err) {
      console.error("⚠️ Failed to poll system status:", err);
      light.textContent =
        "⚠️ Unable to retrieve system status. Please refresh.";
      light.style.color = "#f39c12";
    }
  }

  setInterval(pollStatus, 5000);
  pollStatus();
});

// Render GPT response content
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

  const quotes = data.quotes.join("\n\n");
  const linkedin = data.social_posts.linkedin.join("\n\n");
  const instagram = data.social_posts.instagram.join("\n\n");

  output.innerHTML = `
    <h1>🧠 GPT Interview Summary</h1>
    ${createSection("📌 Compelling Quotes", quotes, quotes)}
    ${createSection("📄 Summary", data.summary, data.summary)}
    ${createSection("💼 LinkedIn Posts", linkedin, linkedin)}
    ${createSection("📸 Instagram Captions", instagram, instagram)}
    ${createSection("📝 Full Transcript", data.transcript, data.transcript)}
  `;
}

// Escape HTML to prevent XSS
function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Copy to clipboard with Toastify
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
