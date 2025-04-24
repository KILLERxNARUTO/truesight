document.addEventListener("DOMContentLoaded", () => {
    const recordBtn = document.getElementById("recordBtn");
    const submitBtn = document.getElementById("submitBtn");
    const userInput = document.getElementById("user_id");
    const form = document.getElementById("loginForm");
    const statusDisplay = document.getElementById("status");
  
    // 🟣 Apply glassmorphic effect to inputs and buttons
    const glassify = (element) => {
      element.style.background = "rgba(255, 255, 255, 0.15)";
      element.style.border = "1px solid rgba(255, 255, 255, 0.3)";
      element.style.backdropFilter = "blur(10px)";
      element.style.borderRadius = "12px";
      element.style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.1)";
      element.style.transition = "all 0.3s ease";
    };
  
    // Apply to form elements
    [recordBtn, submitBtn, userInput].forEach(glassify);
  
    // 🌀 Hover effect for buttons
    [recordBtn, submitBtn].forEach(btn => {
      btn.addEventListener("mouseenter", () => {
        btn.style.transform = "translateY(-2px)";
        btn.style.boxShadow = "0 10px 20px rgba(124, 58, 237, 0.4)";
      });
      btn.addEventListener("mouseleave", () => {
        btn.style.transform = "translateY(0)";
        btn.style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.1)";
      });
    });
  
    // ✨ Input focus effect
    userInput.addEventListener("focus", () => {
      userInput.style.outline = "2px solid #a78bfa";
      userInput.style.boxShadow = "0 0 10px rgba(167, 139, 250, 0.6)";
    });
  
    userInput.addEventListener("blur", () => {
      userInput.style.outline = "none";
      userInput.style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.1)";
    });
  
    // 🟡 Status indicator (message fade-in/out)
    const showStatusMessage = (message, color = "#4b5563") => {
      statusDisplay.textContent = message;
      statusDisplay.style.opacity = "1";
      statusDisplay.style.color = color;
      statusDisplay.style.fontWeight = "500";
      statusDisplay.style.transition = "opacity 0.3s ease-in-out";
  
      setTimeout(() => {
        statusDisplay.style.opacity = "0";
      }, 3000);
    };
  
    // Example call for visual feedback (remove if handled separately)
    recordBtn.addEventListener("click", () => {
      showStatusMessage("🎙️ Voice recording started...", "#8b5cf6");
    });
  
    // Disable submit visually until enabled by backend
    submitBtn.disabled = true;
    submitBtn.style.opacity = "0.6";
  });
  
