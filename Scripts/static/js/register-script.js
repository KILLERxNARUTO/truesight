document.addEventListener("DOMContentLoaded", () => {
  const userIdInput = document.getElementById("user_id");
  const codeInput = document.getElementById("verification_code");
  const codeDisplay = document.getElementById("codeDisplay");
  const recordBtn = document.getElementById("recordBtn");
  const submitBtn = document.getElementById("submitBtn");
  const status = document.getElementById("status");
  const response = document.getElementById("response");
  const getCodeBtn = document.getElementById("getCodeBtn");

  let recordings = [];

  // 🔁 Fetch verification code
  getCodeBtn.addEventListener("click", () => {
    const userId = userIdInput.value.trim();
    if (!userId) return alert("Please enter a user ID first!");

    fetch(`${API_BASE}/api/verification-code?user_id=${encodeURIComponent(userId)}`, {
      credentials: "include" // ✅ Important for session cookie
    })
      .then(res => res.json())
      .then(data => {
        codeDisplay.textContent = `Code: ${data.verification_code}`;
        codeInput.value = data.verification_code;
      })
      .catch(err => {
        alert("Failed to fetch verification code.");
        console.error(err);
      });
  });

  // 🎙️ Record audio samples
  async function recordSample(index) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      let chunks = [];

      mediaRecorder.ondataavailable = e => chunks.push(e.data);
      mediaRecorder.onstop = () => {
        recordings.push(new Blob(chunks, { type: "audio/webm" }));
        status.innerText = `🎤 Sample ${index + 1}/10 recorded`;

        if (index < 9) {
          setTimeout(() => recordSample(index + 1), 1000);
        } else {
          submitBtn.disabled = false;
          status.innerText += " ✅ All samples done!";
        }
      };

      status.innerText = `🔴 Recording sample ${index + 1}/10...`;
      mediaRecorder.start();

      setTimeout(() => {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }, 5000);
    } catch (err) {
      alert("Microphone access denied or failed.");
      console.error(err);
    }
  }

  recordBtn.addEventListener("click", () => {
    recordings = [];
    submitBtn.disabled = true;
    recordSample(0);
  });

  // 📤 Submit registration
  document.getElementById("registerForm").addEventListener("submit", async e => {
    e.preventDefault();

    const code = codeInput.value.trim();
    const userId = userIdInput.value.trim();

    try {
      const verifyRes = await fetch(`${API_BASE}/api/verify-code`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
        credentials: "include" // ✅ Keep session consistent
      });

      const verifyResult = await verifyRes.json();
      if (verifyResult.status !== "success") {
        return alert("Verification failed: " + verifyResult.message);
      }

      const formData = new FormData();
      formData.append("user_id", userId);
      recordings.forEach((blob, index) => {
        formData.append(`audio${index + 1}`, blob, `sample${index + 1}.webm`);
      });

      const res = await fetch(`${API_BASE}/api/register`, {
        method: "POST",
        body: formData,
        credentials: "include" // ✅ Critical for session
      });

      const text = await res.text();
      const result = JSON.parse(text);

      if (result.status === "success") {
        document.getElementById("registerForm").style.display = "none";
        status.style.display = "none";
        response.innerHTML = `✅ ${result.message.replace(/\n/g, "<br>")}`;
      } else {
        response.textContent = result.message;
      }
    } catch (err) {
      console.error("❌ Registration failed:", err);
      alert("⚠️ Server error. Check console for details.");
    }
  });

  // Disable submit initially
  submitBtn.disabled = true;
  submitBtn.style.opacity = "0.6";
});
