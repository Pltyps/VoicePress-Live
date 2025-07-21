document
  .getElementById("uploadForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const file = document.getElementById("videoFile").files[0];
    if (!file) return;

    const status = document.getElementById("uploadStatus");
    status.textContent = "Uploading...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://YOUR-BACKEND-URL/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      status.textContent = response.ok
        ? "✅ " + result.message
        : "❌ " + result.message;

      Toastify({
        text: response.ok ? "✅ Upload complete!" : "❌ Upload failed.",
        duration: 3000,
        gravity: "top",
        position: "right",
        backgroundColor: response.ok ? "#4CAF50" : "#FF4C4C",
      }).showToast();
    } catch (err) {
      console.error(err);
      status.textContent = "❌ Upload failed.";
      Toastify({
        text: "❌ Upload failed. Please try again.",
        duration: 3000,
        gravity: "top",
        position: "right",
        backgroundColor: "#FF4C4C",
      }).showToast();
    }
  });

// Poll server status every 5s
async function pollStatus() {
  try {
    const res = await fetch("https://YOUR-BACKEND-URL/status");
    const data = await res.json();
    const light = document.getElementById("statusLight");

    if (data.status === "processing") {
      light.textContent = "🔴 ON AIR (Processing)";
    } else {
      light.textContent = "🟢 IDLE (Ready)";
    }
  } catch {
    document.getElementById("statusLight").textContent =
      "⚠️ Status check failed";
  }
}

setInterval(pollStatus, 5000);
pollStatus();
