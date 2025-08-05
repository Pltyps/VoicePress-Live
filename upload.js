const BACKEND_URL = "https://voicepress-live-api.onrender.com";
// // Temporarily change for test locally
// const BACKEND_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("videoFile");
  const status = document.getElementById("uploadStatus");
  const spinner = document.getElementById("spinner");
  const progressBar = document.getElementById("uploadProgress");
  const output = document.getElementById("output");
  const fileNameDisplay = document.getElementById("fileName");

  fileInput.addEventListener("change", () => {
    fileNameDisplay.textContent = fileInput.files.length
      ? fileInput.files[0].name
      : "No file chosen";
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

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
        progressBar.value = percent;
      }
    });

    xhr.onload = () => {
      spinner.style.display = "none";
      progressBar.style.display = "none";

      const response = JSON.parse(xhr.responseText);
      if (xhr.status === 200) {
        status.textContent = "✅ Upload complete!";
        renderContent(response);
      } else {
        status.textContent = "❌ Upload failed.";
        output.innerHTML = `<p style="color:red;">${response.error}</p>`;
      }

      fileInput.value = "";
      fileNameDisplay.textContent = "No file chosen";
    };

    xhr.onerror = () => {
      spinner.style.display = "none";
      progressBar.style.display = "none";
      status.textContent = "❌ Upload failed.";
    };

    xhr.send(formData);
  });

  async function pollStatus() {
    const light = document.getElementById("statusLight");

    try {
      const res = await fetch(`${BACKEND_URL}/status`);
      const data = await res.json();

      if (data.status === "processing") {
        light.textContent =
          "🔴 Processing... Generating transcript, summary, quotes, and social posts.";
        light.style.color = "#e74c3c";
        form.querySelector("button").disabled = true;
      } else {
        light.textContent = "🟢 System is ready. You can upload a video file.";
        light.style.color = "#2ecc71";
        form.querySelector("button").disabled = false;
      }
    } catch {
      light.textContent =
        "⚠️ Unable to retrieve system status. Please refresh.";
      light.style.color = "#f39c12";
    }
  }

  setInterval(pollStatus, 5000);
  pollStatus();
});

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

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

///Version 2 without moving the copy button to the frontend and sending the html from the server
// document.addEventListener("DOMContentLoaded", () => {
//   const form = document.getElementById("uploadForm");
//   const fileInput = document.getElementById("videoFile");
//   const status = document.getElementById("uploadStatus");
//   const spinner = document.getElementById("spinner");
//   const progressBar = document.getElementById("uploadProgress");
//   const output = document.getElementById("output");
//   const fileNameDisplay = document.getElementById("fileName");

//   fileInput.addEventListener("change", function () {
//     fileNameDisplay.textContent = fileInput.files.length
//       ? fileInput.files[0].name
//       : "No file chosen";
//   });

//   form.addEventListener("submit", function (e) {
//     e.preventDefault();
//     const file = fileInput.files[0];
//     if (!file) return;

//     const formData = new FormData();
//     formData.append("file", file);

//     const xhr = new XMLHttpRequest();
//     xhr.open("POST", `${BACKEND_URL}/upload`, true);

//     xhr.upload.addEventListener("progress", function (e) {
//       if (e.lengthComputable) {
//         const percent = (e.loaded / e.total) * 100;
//         progressBar.value = percent;
//       }
//     });

//     xhr.onloadstart = () => {
//       spinner.style.display = "block";
//       progressBar.style.display = "block";
//       progressBar.value = 0;
//       status.textContent = "Uploading...";
//     };

//     xhr.onload = () => {
//       spinner.style.display = "none";
//       progressBar.style.display = "none";

//       status.textContent =
//         xhr.status === 200 ? "✅ Upload complete!" : "❌ Upload failed.";

//       Toastify({
//         text: xhr.status === 200 ? "✅ Upload complete!" : "❌ Upload failed.",
//         duration: 3000,
//         gravity: "top",
//         position: "right",
//         backgroundColor: xhr.status === 200 ? "#4CAF50" : "#FF4C4C",
//       }).showToast();

//       if (xhr.status === 200) {
//         output.innerHTML = xhr.responseText;
//       }

//       fileInput.value = "";
//       fileNameDisplay.textContent = "No file chosen";
//     };

//     xhr.onerror = () => {
//       spinner.style.display = "none";
//       progressBar.style.display = "none";
//       status.textContent = "❌ Upload failed.";
//       Toastify({
//         text: "❌ Upload failed. Please try again.",
//         duration: 3000,
//         gravity: "top",
//         position: "right",
//         backgroundColor: "#FF4C4C",
//       }).showToast();
//     };

//     xhr.send(formData);
//   });

//   // ✅ Poll server status
//   async function pollStatus() {
//     const light = document.getElementById("statusLight");

//     try {
//       const res = await fetch(`${BACKEND_URL}/status`);
//       const data = await res.json();

//       if (data.status === "processing") {
//         light.textContent =
//           "🔴 Processing... Generating transcript, summary, quotes, and social posts.";
//         light.style.color = "#e74c3c";
//         form.querySelector("button").disabled = true;
//       } else {
//         light.textContent = "🟢 System is ready. You can upload a video file.";
//         light.style.color = "#2ecc71";
//         form.querySelector("button").disabled = false;
//       }
//     } catch {
//       light.textContent =
//         "⚠️ Unable to retrieve system status. Please refresh.";
//       light.style.color = "#f39c12";
//     }
//   }

//   setInterval(pollStatus, 5000);
//   pollStatus();
// });

//////// Orignial
// document
//   .getElementById("uploadForm")
//   .addEventListener("submit", async function (e) {
//     e.preventDefault();
//     const file = document.getElementById("videoFile").files[0];
//     if (!file) return;

//     const spinner = document.getElementById("spinner");
//     const status = document.getElementById("uploadStatus");
//     status.textContent = "Uploading...";
//     document.getElementById("spinner").style.display = "block";
//     const formData = new FormData();
//     formData.append("file", file);

//     try {
//       // const response = await fetch(`${BACKEND_URL}/upload`, {
//       //   method: "POST",
//       //   body: formData,
//       // });

//       // const html = await response.text(); // Expecting HTML, not JSON
//       // document.getElementById("output").innerHTML = html;

//       // status.textContent = response.ok
//       //   ? "✅ Upload complete!"
//       //   : "❌ Upload failed.";

//       // Toastify({
//       //   text: response.ok ? "✅ Upload complete!" : "❌ Upload failed.",
//       //   duration: 3000,
//       //   gravity: "top",
//       //   position: "right",
//       //   backgroundColor: response.ok ? "#4CAF50" : "#FF4C4C",
//       // }).showToast();
//       const xhr = new XMLHttpRequest();
//       xhr.open("POST", `${BACKEND_URL}/upload`, true);

//       xhr.upload.addEventListener("progress", function (e) {
//         if (e.lengthComputable) {
//           const percent = (e.loaded / e.total) * 100;
//           document.getElementById("uploadProgress").value = percent;
//         }
//       });

//       xhr.onloadstart = () => {
//         document.getElementById("uploadProgress").style.display = "block";
//         document.getElementById("uploadProgress").value = 0;
//         status.textContent = "Uploading...";
//       };

//       xhr.onload = () => {
//         spinner.style.display = "none";
//         document.getElementById("uploadProgress").style.display = "none";
//         document.getElementById("spinner").style.display = "none";

//         const html = xhr.responseText;
//         document.getElementById("output").innerHTML = html;

//         status.textContent =
//           xhr.status === 200 ? "✅ Upload complete!" : "❌ Upload failed.";

//         Toastify({
//           text:
//             xhr.status === 200 ? "✅ Upload complete!" : "❌ Upload failed.",
//           duration: 3000,
//           gravity: "top",
//           position: "right",
//           backgroundColor: xhr.status === 200 ? "#4CAF50" : "#FF4C4C",
//         }).showToast();

//         document.getElementById("videoFile").value = ""; // Reset input
//       };

//       xhr.onerror = () => {
//         spinner.style.display = "none";
//         document.getElementById("uploadProgress").style.display = "none";
//         status.textContent = "❌ Upload failed.";
//         Toastify({
//           text: "❌ Upload failed. Please try again.",
//           duration: 3000,
//           gravity: "top",
//           position: "right",
//           backgroundColor: "#FF4C4C",
//         }).showToast();
//       };

//       xhr.send(formData);
//     } catch (err) {
//       console.error(err);
//       status.textContent = "❌ Upload failed.";
//       Toastify({
//         text: "❌ Upload failed. Please try again.",
//         duration: 3000,
//         gravity: "top",
//         position: "right",
//         backgroundColor: "#FF4C4C",
//       }).showToast();
//     }
//   });

// // Poll server status every 5s
// async function pollStatus() {
//   try {
//     const res = await fetch(`${BACKEND_URL}/status`);
//     const data = await res.json();
//     const light = document.getElementById("statusLight");

//     if (data.status === "processing") {
//       light.textContent = "🔴 ON AIR (Processing)";
//     } else {
//       light.textContent = "🟢 IDLE (Ready)";
//     }
//   } catch {
//     document.getElementById("statusLight").textContent =
//       "⚠️ Status check failed";
//   }
// }

// setInterval(pollStatus, 5000);
// pollStatus();
