
// ======================================================
//  SCRIPT READY CHECK
// ======================================================
console.log("✅ Nieuw script.js geladen (cross-browser compatible)");


// ======================================================
//  VARIABELEN
// ======================================================
let mediaRecorder;
let bufferChunks = [];
let isSpeaking = false;
let noiseFloorRms = 0.005;

let audioContext;
let analyser;
let source;
let lastSpeechTime = Date.now();

let isPaused = false;
let intervalId = null;

const micStatusElement = document.getElementById("micStatus");
let micStatusState = "idle";

function escapeHtml(text) {
  const replacements = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return String(text).replace(/[&<>"']/g, (char) => replacements[char] || char);
}

function setMicStatus(state, detail = "") {
  if (!micStatusElement) return;

  let label = "🎙️ Microphone idle";
  let fallbackDetail = "Press start to begin calibration";

  if (state === "calibrating") {
    label = "🎚️ Calibrating ambient noise…";
    fallbackDetail = "Stay silent for a moment";
  } else if (state === "listening") {
    label = "👂 Listening";
    fallbackDetail = "Waiting for speech";
  } else if (state === "speaking") {
    label = "🗣️ Speech detected";
    fallbackDetail = "";
  } else if (state === "error") {
    label = "❌ Microphone unavailable";
    fallbackDetail = "Grant microphone access and try again";
  }

  const detailText = detail || fallbackDetail;

  if (
    state === micStatusState &&
    micStatusElement.dataset.detail === detailText
  ) {
    return;
  }

  micStatusElement.classList.remove(
    "idle",
    "calibrating",
    "listening",
    "speaking",
    "error"
  );

  micStatusElement.classList.add(state);
  micStatusState = state;
  micStatusElement.dataset.detail = detailText;

  const safeDetail = detailText ? escapeHtml(detailText) : "";
  micStatusElement.innerHTML = detailText
    ? `${label}<small>${safeDetail}</small>`
    : label;
}

setMicStatus("idle");


// ======================================================
//  SPREKEN-DETECTIE
// ======================================================
async function setupAudioDetection(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  const floatTimeData = new Float32Array(analyser.fftSize);
  const byteTimeData = new Uint8Array(analyser.fftSize);

  intervalId = setInterval(() => {
    let sumSquares = 0;

    if (typeof analyser.getFloatTimeDomainData === "function") {
      analyser.getFloatTimeDomainData(floatTimeData);
      for (let i = 0; i < floatTimeData.length; i++) {
        const sample = floatTimeData[i];
        sumSquares += sample * sample;
      }
    } else {
      analyser.getByteTimeDomainData(byteTimeData);
      for (let i = 0; i < byteTimeData.length; i++) {
        const centeredSample = (byteTimeData[i] - 128) / 128;
        sumSquares += centeredSample * centeredSample;
      }
    }

    const rms = Math.sqrt(sumSquares / analyser.fftSize);

    const silenceThreshold = Math.max(noiseFloorRms * 1.8, 0.006);
    isSpeaking = rms > silenceThreshold;

    if (isSpeaking) {
      lastSpeechTime = Date.now();
    } else {
      noiseFloorRms = Math.max(0.001, noiseFloorRms * 0.95 + rms * 0.05);
    }
  }, 150);
}


// ======================================================
//  TTS VIA BACKEND
// ======================================================
async function spreekVertaling(text, lang) {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("lang", lang);
  formData.append("speak", "true");

  const response = await fetch("/api/speak", { method: "POST", body: formData });
  const blob = await response.blob();

  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}


// ======================================================
//  MEDIARECORDER — CROSS-BROWSER FALLBACK
// ======================================================
function getRecorderOptions() {
  let options = { mimeType: "audio/webm;codecs=opus" };

  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    console.warn("⚠️ webm/opus niet ondersteund → overschakelen naar mp4/aac");
    options = { mimeType: "audio/mp4" };
  }

  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    console.warn("⚠️ mp4 niet ondersteund → browser kiest automatisch");
    options = {};
  }

  return options;
}


// ======================================================
//  START KNOP
// ======================================================
document.getElementById("start").onclick = async () => {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }

  bufferChunks = [];
  isSpeaking = false;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    console.log("🎙️ Microfoon toestemming OK");
    await setupAudioDetection(stream);

    const options = getRecorderOptions();
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.start(5000); // elke 5 sec fragment

    mediaRecorder.ondataavailable = async (event) => {␊
      const now = Date.now();␊
      const stilte = now - lastSpeechTime;␊
␊
      if (stilte < 3000) {␊
        bufferChunks.push(event.data);
      } else if (bufferChunks.length > 0) {
        const blob = new Blob(bufferChunks, { type: event.data.type });
        bufferChunks = [];

        const formData = new FormData();
        formData.append("audio", blob, "spraak." + event.data.type.split("/")[1]);
        formData.append("from", document.getElementById("sourceLanguage").value);
        formData.append("to", document.getElementById("languageSelect").value);
        formData.append(
          "textOnly",
          document.getElementById("textOnly").checked ? "true" : "false"
        );

        const unsupportedByDeepL = [
          "sw", "am", "mg", "lingala", "kikongo",
          "tshiluba", "baloué", "dioula"
        ];

        if (unsupportedByDeepL.includes(document.getElementById("languageSelect").value)) {
          alert("⚠️ This language isn't supported by DeepL. AI will translate instead.");
        }

        const response = await fetch("/api/translate", { method: "POST", body: formData });
        const data = await response.json();

        document.getElementById("transcript").innerHTML += `<p>${data.original}</p>`;
        document.getElementById("orig").innerHTML       += `<p>${data.original}</p>`;
        document.getElementById("trans").innerHTML      += `<p>${data.translation}</p>`;

        if (!document.getElementById("textOnly").checked) {
          spreekVertaling(data.translation, document.getElementById("languageSelect").value);
        }
      }
    };

    // Knoppen togglen
    document.getElementById("start").disabled = true;
    document.getElementById("pause").disabled = false;
    document.getElementById("stop").disabled = false;
  
    } catch (err) {
      alert("❌ Microfoon werkt niet: " + err.message);
      console.error("Microfoonfout:", err);
      setMicStatus("error", err.message || "");
    }
  };


// ======================================================
//  PAUSE KNOP
// ======================================================
document.getElementById("pause").onclick = () => {
  if (!mediaRecorder) return;

  if (!isPaused && mediaRecorder.state === "recording") {
    mediaRecorder.pause();
    isPaused = true;
    document.getElementById("pause").innerText = "▶️ continue ▶️";
  } else {
    mediaRecorder.resume();
    isPaused = false;
    document.getElementById("pause").innerText = "⏸️ pause ⏸️";
  }
};


// ======================================================
//  STOP KNOP
// ======================================================
document.getElementById("stop").onclick = () => {
  clearInterval(intervalId);
  intervalId = null;

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  document.getElementById("start").disabled = false;
  document.getElementById("pause").disabled = true;
  document.getElementById("stop").disabled = true;
  document.getElementById("pause").innerText = "⏸️ pause ⏸️ ";

  isPaused = false;
  noiseFloorRms = 0.005;
  bufferChunks = [];
  isSpeaking = false;
  setMicStatus("idle");
};
