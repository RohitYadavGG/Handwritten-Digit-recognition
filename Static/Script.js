// ============================================
// HEADER SCROLL HIDE/SHOW
// ============================================
let lastScrollY = window.scrollY;
const header = document.querySelector('.header');

window.addEventListener('scroll', () => {
  const currentScrollY = window.scrollY;

  // Hide/Show logic based on scroll direction
  if (currentScrollY <= 50) {
    // Always show when near the top
    header.classList.remove('hidden');
  } else if (currentScrollY > lastScrollY) {
    // Scrolling down -> hide the navbar
    header.classList.add('hidden');
  } else {
    // Scrolling up -> show the navbar
    header.classList.remove('hidden');
  }
  
  lastScrollY = currentScrollY;
});

// ============================================
// DRAWING CANVAS
// ============================================
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
let canvasImageData = null; // Store canvas state for theme changes

// Get canvas background color - ALWAYS WHITE to match MNIST training
function getCanvasBackgroundColor() {
  return "#ffffff"; // Always white for MNIST compatibility
}

// Get stroke color - ALWAYS BLACK to match MNIST training data
function getStrokeColor() {
  return "#000000"; // Always black for MNIST compatibility
}

// Initialize canvas for 28x28 drawing
function initializeCanvas() {
  // Canvas is 28x28, displayed at 280x280 via CSS for visibility
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 2;  // Thicker line for visibility on small canvas

  const bgColor = getCanvasBackgroundColor();
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, 28, 28);
}

// Redraw canvas background when theme changes
function redrawCanvasBackground() {
  // Reset canvas
  ctx.clearRect(0, 0, 28, 28);

  // Fill with new theme background color
  const bgColor = getCanvasBackgroundColor();
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, 28, 28);
}

initializeCanvas();

// Drawing functions
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch support
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouch);
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  // Scale coordinates from displayed 280x280 to actual 28x28
  const scale = canvas.width / rect.width;
  const x = (e.clientX - rect.left) * scale;
  const y = (e.clientY - rect.top) * scale;
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  // Scale coordinates from displayed 280x280 to actual 28x28
  const scale = canvas.width / rect.width;
  const x = (e.clientX - rect.left) * scale;
  const y = (e.clientY - rect.top) * scale;
  ctx.strokeStyle = getStrokeColor();
  ctx.lineTo(x, y);
  ctx.stroke();
}

function handleTouch(e) {
  const touch = e.touches[0];
  const mouseEvent = new MouseEvent(e.type === "touchstart" ? "mousedown" : "mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY,
  });
  canvas.dispatchEvent(mouseEvent);
}

function stopDrawing() {
  isDrawing = false;
  ctx.closePath();
}

// Clear canvas
document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.fillStyle = getCanvasBackgroundColor();
  ctx.fillRect(0, 0, 28, 28);
  document.getElementById("drawPrediction").style.display = "none";
  updateCanvasButtonState();
});

// ============================================
// PREDICT FROM DRAWING
// ============================================
document.getElementById("predictBtn").addEventListener("click", async () => {
  // Check if canvas is empty - with fallback
  let isEmpty = false;
  if (typeof isCanvasEmpty === 'function') {
    isEmpty = isCanvasEmpty();
  }

  if (isEmpty) {
    alert("Please draw a digit first!");
    return;
  }

  showSpinner(true);
  try {
    const imageData = canvas.toDataURL("image/png");
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) throw new Error("Prediction failed");
    const result = await response.json();

    console.log("[DEBUG] Prediction received:", result);

    // Update UI with prediction
    document.getElementById("drawResultValue").textContent = result.prediction;
    document.getElementById("drawConfidence").textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    const confidenceBar = document.getElementById("drawConfidenceBar");
    if (confidenceBar) {
      confidenceBar.style.width = '0%'; // Reset first for animation
      setTimeout(() => {
        confidenceBar.style.width = `${(result.confidence * 100).toFixed(1)}%`;
      }, 50);
    }
    
    document.getElementById("drawPrediction").style.display = "block";
  } catch (error) {
    console.error("[ERROR] Prediction error:", error);
    alert("Error making prediction: " + error.message);
  } finally {
    showSpinner(false);
  }
});

// ============================================
// IMAGE UPLOAD
// ============================================
const uploadBox = document.getElementById("uploadBox");
const imageInput = document.getElementById("imageInput");
const uploadPreview = document.getElementById("uploadPreview");
const previewImg = document.getElementById("previewImg");
const uploadPredictBtn = document.getElementById("uploadPredictBtn");
const removeImageBtn = document.getElementById("removeImageBtn");

uploadBox.addEventListener("click", () => imageInput.click());

uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.classList.add("dragover");
});

uploadBox.addEventListener("dragleave", () => {
  uploadBox.classList.remove("dragover");
});

uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadBox.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) handleImageUpload(files[0]);
});

imageInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleImageUpload(e.target.files[0]);
});

function handleImageUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadPreview.style.display = "flex";
    uploadBox.style.display = "none";
    uploadPredictBtn.style.display = "inline-block";
    removeImageBtn.style.display = "inline-block";
    
    // Add the has-image state to trigger the side-by-side grid layout
    const container = document.querySelector(".upload-container");
    if (container) container.classList.add("has-image");
    
    const samplesContainer = document.getElementById("samplesContainer");
    if (samplesContainer) samplesContainer.style.display = "none";
    document.getElementById("uploadPrediction").style.display = "none";
  };
  reader.readAsDataURL(file);
}

removeImageBtn.addEventListener("click", () => {
  uploadPreview.style.display = "none";
  uploadBox.style.display = "block";
  uploadPredictBtn.style.display = "none";
  removeImageBtn.style.display = "none";
  
  // Remove the has-image state to revert to centered layout
  const container = document.querySelector(".upload-container");
  if (container) container.classList.remove("has-image");
  
  const samplesContainer = document.getElementById("samplesContainer");
  if (samplesContainer) samplesContainer.style.display = "block";
  document.getElementById("uploadPrediction").style.display = "none";
  imageInput.value = "";
});

// ============================================
// PREDICT FROM UPLOAD
// ============================================
uploadPredictBtn.addEventListener("click", async () => {
  showSpinner(true);
  try {
    // Get the file from input
    const file = imageInput.files[0];
    if (!file) {
      alert("No file selected");
      return;
    }

    // Create FormData with the file
    const formData = new FormData();
    formData.append('file', file);

    // Send to /predict-image endpoint (designed for file uploads)
    const response = await fetch("/predict-image", {
      method: "POST",
      body: formData,
      // Do NOT set Content-Type header - browser will set it with boundary
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Prediction failed');
    }

    const result = await response.json();

    console.log("[DEBUG] Upload prediction received:", result);

    // Update UI with prediction
    document.getElementById("uploadResultValue").textContent = result.prediction;
    document.getElementById("uploadConfidence").textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    const confidenceBar = document.getElementById("uploadConfidenceBar");
    if (confidenceBar) {
      confidenceBar.style.width = '0%'; // Reset first for animation
      setTimeout(() => {
        confidenceBar.style.width = `${(result.confidence * 100).toFixed(1)}%`;
      }, 50);
    }
    
    document.getElementById("uploadPrediction").style.display = "block";

    // Also show all probabilities in console
    console.log('Full predictions:', result.probabilities);
  } catch (error) {
    console.error("[ERROR] Upload prediction error:", error);
    alert(`Error making prediction: ${error.message}`);
  } finally {
    showSpinner(false);
  }
});

// ============================================
// SAMPLE IMAGES
// ============================================
document.querySelectorAll(".btn-sample").forEach((btn) => {
  btn.addEventListener("click", async () => {
    showSpinner(true);
    try {
      const sampleNum = btn.getAttribute("data-sample");
      const response = await fetch(`/sample/${sampleNum}`);
      if (!response.ok) throw new Error("Failed to load sample");
      const data = await response.json();
      previewImg.src = data.image;
      uploadPreview.style.display = "flex";
      uploadBox.style.display = "none";
      uploadPredictBtn.style.display = "inline-block";
      removeImageBtn.style.display = "inline-block";
      
      const container = document.querySelector(".upload-container");
      if (container) container.classList.add("has-image");
      
      const samplesContainer = document.getElementById("samplesContainer");
      if (samplesContainer) samplesContainer.style.display = "none";
      document.getElementById("uploadPrediction").style.display = "none";
    } catch (error) {
      console.error("Error:", error);
      alert("Error loading sample image.");
    } finally {
      showSpinner(false);
    }
  });
});

// ============================================
// SPINNER
// ============================================
function showSpinner(show) {
  const spinner = document.getElementById("spinner");
  spinner.style.display = show ? "block" : "none";
}

// ============================================
// SMOOTH SCROLL FOR NAV LINKS
// ============================================
document.querySelectorAll("a[href^='#']").forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    const href = this.getAttribute("href");
    if (href !== "#") {
      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  });
});

// ============================================
// TYPEWRITER EFFECT
// ============================================
document.addEventListener("DOMContentLoaded", () => {
  const typedTextSpan = document.getElementById('typed-text');
  if (typedTextSpan) {
    const originalHTML = `Handwritten <span class="hero-digit">Digit</span><br>Recognition 0 to 9`;
    let i = 0;
    let isTag = false;
    let text = '';

    function typeWriter() {
      if (i < originalHTML.length) {
        if (originalHTML.charAt(i) === '<') {
          isTag = true;
        }

        text += originalHTML.charAt(i);

        if (originalHTML.charAt(i) === '>') {
          isTag = false;
        }

        if (isTag) {
          i++;
          typeWriter(); // Skip delays for HTML tags
        } else {
          typedTextSpan.innerHTML = text;
          i++;
          // Random typing speed between 30ms and 80ms
          setTimeout(typeWriter, 30 + Math.random() * 50);
        }
      } else {
        // Typing complete! Reveal the hidden UI elements.
        const cursor = document.querySelector('.typewriter-cursor');
        if (cursor) cursor.classList.add('typing-complete');

        document.querySelectorAll('.hide-initial').forEach(el => {
          el.classList.add('reveal-post-typing');
          setTimeout(() => {
            el.classList.remove('hide-initial');
            el.classList.remove('reveal-post-typing');
          }, 2000);
        });

        // Reveal Vanta Globe 0.4s after the buttons appear
        setTimeout(() => {
          if (typeof window.revealVantaGlobe === 'function') {
            window.revealVantaGlobe();
          }
        }, 400);

        // Loop: alternate between "0 to 9" and "0 - 9"
        const staticPart = `Handwritten <span class="hero-digit">Digit</span><br>Recognition `;
        const loopTexts = ['0 to 9', '0 - 9'];
        let currentIndex = 0; // points to what is currently displayed

        function eraseLoop() {
          const currentText = loopTexts[currentIndex];
          const nextIndex = (currentIndex + 1) % loopTexts.length;
          const nextText = loopTexts[nextIndex];
          let current = currentText;

          function eraseChar() {
            if (current.length > 0) {
              current = current.slice(0, -1);
              typedTextSpan.innerHTML = staticPart + current;
              setTimeout(eraseChar, 60 + Math.random() * 40);
            } else {
              retypeChar();
            }
          }

          function retypeChar() {
            if (current.length < nextText.length) {
              current = nextText.slice(0, current.length + 1);
              typedTextSpan.innerHTML = staticPart + current;
              setTimeout(retypeChar, 80);
            } else {
              currentIndex = nextIndex; // update tracker after full retype
              setTimeout(eraseLoop, 1500);
            }
          }

          eraseChar();
        }

        // Start the loop after a 2s pause following the initial type
        setTimeout(eraseLoop, 2000);
      }
    }

    // Start typing after a short delay
    setTimeout(typeWriter, 500);
  }
});
