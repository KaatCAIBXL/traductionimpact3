// ======================================================
//  SCRIPT READY CHECK
// ======================================================
console.log("✅ Nieuw script.js geladen (cross-browser compatible)");


// ======================================================
//  VARIABELEN
// ======================================================
let mediaRecorder;
let bufferChunks = [];
let bufferedDurationMs = 0;
let isSpeaking = false;
let noiseFloorRms = 0.005;

let audioContext;
let analyser;
let source;
let lastSpeechTime = Date.now();

let isPaused = false;
let intervalId = null;
let activeStream = null;
let previousSpeakingState = false;

const micStatusElement = document.getElementById("micStatus");
const startButton = document.getElementById("start");
const pauseButton = document.getElementById("pause");
const stopButton = document.getElementById("stop");
const textOnlyCheckbox = document.getElementById("textOnly");
const sourceLanguageSelect = document.getElementById("sourceLanguage");
const targetLanguageSelect = document.getElementById("languageSelect");
const recognizedContainer = document.getElementById("orig");
const correctedContainer = document.getElementById("transcript");
const translationContainer = document.getElementById("trans");
let micStatusState = "idle";

const CHUNK_INTERVAL_MS = 1500;
const SILENCE_FLUSH_MS = 1200;
const MAX_BUFFER_MS = 6000;
let sessionSegments = [];

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

function renderLatestSegments() {
  const recent = sessionSegments.slice(-2);

  if (recognizedContainer) {
    recognizedContainer.innerHTML = recent
      .map((segment) => `<p>${escapeHtml(segment.recognized || "")}</p>`)
      .join("");
  }

  if (correctedContainer) {
    correctedContainer.innerHTML = recent
      .map((segment) => `<p>${escapeHtml(segment.corrected || "")}</p>`)
      .join("");
  }

  if (translationContainer) {
    translationContainer.innerHTML = recent
      .map((segment) => `<p>${escapeHtml(segment.translation || "")}</p>`)
      .join("");
  }
}

renderLatestSegments();

function stopActiveStream() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }
}

function releaseAudioResources() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }

  if (audioContext && typeof audioContext.close === "function") {
    audioContext.close().catch(() => {});
  }

  audioContext = null;
  analyser = null;
  source = null;
  previousSpeakingState = false;
  stopActiveStream();
}

if (micStatusElement && startButton) {
  micStatusElement.addEventListener("click", () => {
    if (!startButton.disabled) {
      startButton.click();
    }
  });

  micStatusElement.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!startButton.disabled) {
        startButton.click();
      }
    }
  });
}


// ======================================================
//  SPREKEN-DETECTIE
// ======================================================
async function setupAudioDetection(stream) {
  setMicStatus("calibrating");
  previousSpeakingState = false;
  noiseFloorRms = 0.005;
  lastSpeechTime = Date.now();

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  const floatTimeData = new Float32Array(analyser.fftSize);
  const byteTimeData = new Uint8Array(analyser.fftSize);
  const calibrationStart = Date.now();
  let calibrationDone = false;

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

    if (!calibrationDone && Date.now() - calibrationStart > 1200) {
      calibrationDone = true;
      setMicStatus("listening");
    }

    if (calibrationDone) {
      if (isSpeaking && !previousSpeakingState) {
        setMicStatus("speaking");
      } else if (!isSpeaking && previousSpeakingState) {
        setMicStatus("listening");
      }
    }

    previousSpeakingState = isSpeaking;
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
if (startButton) {
  startButton.onclick = async () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    releaseAudioResources();
    bufferChunks = [];
    bufferedDurationMs = 0;
    isSpeaking = false;
    sessionSegments = [];
    renderLatestSegments();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const message = "Browser ondersteunt geen microfoon opname.";
      alert("❌ " + message);
      setMicStatus("error", message);
      return;
    }

    try {
      setMicStatus("calibrating");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      activeStream = stream;

      console.log("🎙️ Microfoon toestemming OK");
      await setupAudioDetection(stream);

      const options = getRecorderOptions();
      mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorder.start(CHUNK_INTERVAL_MS); // kortere fragmenten voor snellere transcriptie

      mediaRecorder.ondataavailable = async (event) => {
        const now = Date.now();
        const stilte = now - lastSpeechTime;

        if (event.data && event.data.size) {
          bufferChunks.push(event.data);
          bufferedDurationMs += CHUNK_INTERVAL_MS;
        }

        const shouldFlush =
          bufferChunks.length > 0 &&
          (stilte >= SILENCE_FLUSH_MS || bufferedDurationMs >= MAX_BUFFER_MS);

        if (!shouldFlush) {
          return;
        }

        const rawMimeType = event.data.type || "audio/webm";
        const cleanMimeType = rawMimeType.split(";")[0].trim() || "audio/webm";
        const blob = new Blob(bufferChunks, { type: cleanMimeType });
        bufferChunks = [];
        bufferedDurationMs = 0;

        const extension = cleanMimeType.includes("/")
          ? cleanMimeType.split("/")[1].trim() || "webm"
          : "webm";

        const formData = new FormData();
        formData.append("audio", blob, `spraak.${extension}`);
        formData.append("from", sourceLanguageSelect.value);
        formData.append("to", targetLanguageSelect.value);
        formData.append("textOnly", textOnlyCheckbox.checked ? "true" : "false");

        const unsupportedByDeepL = [
          "sw", "am", "mg", "lingala", "kikongo",
          "tshiluba", "baloué", "dioula"
        ];

        if (unsupportedByDeepL.includes(targetLanguageSelect.value)) {
          alert("⚠️ This language isn't supported by DeepL. AI will translate instead.");
        }

        const response = await fetch("/api/translate", { method: "POST", body: formData });
        const data = await response.json();

        if (data.error) {
          console.error("Vertaalfout:", data.error);
          return;
        }

        const segment = {
          recognized: data.recognized || "",
          corrected: data.corrected || data.recognized || "",
          translation: data.translation || "",
        };

        sessionSegments.push(segment);
        renderLatestSegments();

        if (!textOnlyCheckbox.checked && segment.translation) {
          spreekVertaling(segment.translation, targetLanguageSelect.value);
        }
      };

      mediaRecorder.addEventListener("stop", () => {
        releaseAudioResources();
        mediaRecorder = null;
      });

      mediaRecorder.addEventListener("error", (event) => {
        console.error("Recorder error", event.error);
        setMicStatus("error", event.error?.message || "Recorder error");
        releaseAudioResources();
        if (startButton) startButton.disabled = false;
        if (pauseButton) {
          pauseButton.disabled = true;
          pauseButton.innerText = "⏸️ pause ⏸️";
        }
        if (stopButton) stopButton.disabled = true;
      });

      // Knoppen togglen
      startButton.disabled = true;
      if (pauseButton) pauseButton.disabled = false;
      if (stopButton) stopButton.disabled = false;
    } catch (err) {
      alert("❌ Microfoon werkt niet: " + err.message);
      console.error("Microfoonfout:", err);
      setMicStatus("error", err.message || "");
      releaseAudioResources();
    }
  };
}


// ======================================================
//  PAUSE KNOP
// ======================================================
if (pauseButton) {
  pauseButton.onclick = () => {
    if (!mediaRecorder) return;

    if (!isPaused && mediaRecorder.state === "recording") {
      mediaRecorder.pause();
      isPaused = true;
      pauseButton.innerText = "▶️ continue ▶️";
    } else if (mediaRecorder.state === "paused") {
      mediaRecorder.resume();
      isPaused = false;
      pauseButton.innerText = "⏸️ pause ⏸️";
    }
  };
}


// ======================================================
//  STOP KNOP
// ======================================================
if (stopButton) {
  stopButton.onclick = () => {
    releaseAudioResources();

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }

    if (startButton) startButton.disabled = false;
    if (pauseButton) {
      pauseButton.disabled = true;
      pauseButton.innerText = "⏸️ pause ⏸️";
    }
    stopButton.disabled = true;

    isPaused = false;
    noiseFloorRms = 0.005;
    bufferChunks = [];
    bufferedDurationMs = 0;
    isSpeaking = false;
    downloadSessionDocument();
    setMicStatus("idle");
  };
}

function downloadSessionDocument() {
  if (!sessionSegments.length) {
    return;
  }

  const parts = sessionSegments.map((segment, index) => {
    const nummer = index + 1;
    return [
      `Deel ${nummer}`,
      `Herkenning: ${segment.recognized || ""}`,
      `Correctie: ${segment.corrected || ""}`,
      `Vertaling: ${segment.translation || ""}`,
    ].join("\n");
  });

  const content = parts.join("\n\n");
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `transcriptie-${new Date()
    .toISOString()
    .replace(/[:.]/g, "-")}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

