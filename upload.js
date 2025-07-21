document
  .getElementById("uploadForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();
    const file = document.getElementById("videoFile").files[0];
    if (!file) return;

    const status = document.getElementById("status");
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
        ? "Upload successful! " + result.message
        : "Error: " + result.message;
    } catch (err) {
      console.error(err);
      status.textContent = "Upload failed.";
    }
  });
