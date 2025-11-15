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
let recorderOptions = null;
let pendingRecorderRestart = false;
let isRestartingRecorder = false;

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
  recorderOptions = null;
  pendingRecorderRestart = false;
  isRestartingRecorder = false;
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
function sanitizeMimeType(rawType) {
  if (typeof rawType !== "string") {
    return "";
  }

  const baseType = rawType.split(";")[0].trim().toLowerCase();
  return baseType;
}

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

async function sniffMimeTypeFromChunks(chunks) {
  const sampleChunk = chunks.find((chunk) => chunk && chunk.size);

  if (!sampleChunk) {
    return "";
  }

  try {
    const header = new Uint8Array(await sampleChunk.slice(0, 16).arrayBuffer());

    if (
      header.length >= 4 &&
      header[0] === 0x1a &&
      header[1] === 0x45 &&
      header[2] === 0xdf &&
      header[3] === 0xa3
    ) {
      return "audio/webm";
    }

    if (
      header.length >= 12 &&
      header[4] === 0x66 &&
      header[5] === 0x74 &&
      header[6] === 0x79 &&
      header[7] === 0x70
    ) {
      return "audio/mp4";
    }

    if (
      header.length >= 4 &&
      header[0] === 0x4f &&
      header[1] === 0x67 &&
      header[2] === 0x67 &&
      header[3] === 0x53
    ) {
      return "audio/ogg";
    }

    if (
      header.length >= 12 &&
      header[0] === 0x52 &&
      header[1] === 0x49 &&
      header[2] === 0x46 &&
      header[3] === 0x46 &&
      header[8] === 0x57 &&
      header[9] === 0x41 &&
      header[10] === 0x56 &&
      header[11] === 0x45
    ) {
      return "audio/wav";
    }
  } catch (error) {
    console.warn("Kon bestandskop niet inspecteren:", error);
  }

  return "";
}

async function resolveMimeType(chunks, fallbackTypes = []) {
  const chunkWithType = chunks.find(
    (chunk) => chunk && typeof chunk.type === "string" && chunk.type
  );
                                                            

  if (chunkWithType) {
    const clean = sanitizeMimeType(chunkWithType.type);
    if (clean) {
      return clean;
    }
  }

  const sniffed = await sniffMimeTypeFromChunks(chunks);
  if (sniffed) {
    return sniffed;
  }

  for (const rawType of fallbackTypes) {
    const clean = sanitizeMimeType(rawType);
    if (clean) {
      return clean;
    }
  }

  return "audio/webm";
}

const MIME_EXTENSION_MAP = {
  "audio/webm": "webm",
  "video/webm": "webm",
  "audio/ogg": "ogg",
  "video/ogg": "ogg",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/mp4": "mp4",
  "video/mp4": "mp4",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/aac": "aac",
  "audio/3gpp": "3gp",
  "audio/3gpp2": "3g2",
};

function mimeTypeToExtension(mimeType) {
  const clean = sanitizeMimeType(mimeType);

  if (clean && MIME_EXTENSION_MAP[clean]) {
    return MIME_EXTENSION_MAP[clean];
  }

  if (clean && clean.includes("/")) {
    const parts = clean.split("/");
    const candidate = parts[1].trim();
    if (candidate) {
      return candidate;
    }
  }

  return "webm";
}

function initializeMediaRecorder(stream, optionsOverride) {
  if (!stream) {
    return null;
  }

  const options = optionsOverride || recorderOptions || getRecorderOptions();
  recorderOptions = options;

  const recorder = new MediaRecorder(stream, options);

  const handleStop = () => {
    const shouldRestart = pendingRecorderRestart;
    pendingRecorderRestart = false;
    isRestartingRecorder = false;
    mediaRecorder = null;

    if (shouldRestart && stream.active) {
      bufferChunks = [];
      bufferedDurationMs = 0;
      try {
        initializeMediaRecorder(stream, recorderOptions);
        if (pauseButton) {
          pauseButton.disabled = false;
          pauseButton.innerText = "⏸️ pause ⏸️";
        }
        if (stopButton) {
          stopButton.disabled = false;
        }
      } catch (error) {
        console.error("Kon MediaRecorder niet herstarten:", error);
        setMicStatus("error", "Kon opname niet herstarten");
        releaseAudioResources();
        if (startButton) startButton.disabled = false;
      }
      return;
    }

    releaseAudioResources();
    if (startButton) startButton.disabled = false;
    if (pauseButton) {
      pauseButton.disabled = true;
      pauseButton.innerText = "⏸️ pause ⏸️";
    }
    if (stopButton) stopButton.disabled = true;
    isPaused = false;
    setMicStatus("idle");
  };

  const handleError = (event) => {
    if (pendingRecorderRestart) {
      return;
    }

    console.error("Recorder error", event.error);
    setMicStatus("error", event.error?.message || "Recorder error");
    releaseAudioResources();
    mediaRecorder = null;
    if (startButton) startButton.disabled = false;
    if (pauseButton) {
      pauseButton.disabled = true;
      pauseButton.innerText = "⏸️ pause ⏸️";
    }
    if (stopButton) stopButton.disabled = true;
  };

  recorder.addEventListener("stop", handleStop);
  recorder.addEventListener("error", handleError);
  recorder.addEventListener("dataavailable", handleDataAvailable);

  recorder.start(CHUNK_INTERVAL_MS);
  mediaRecorder = recorder;
  return recorder;
}

function scheduleRecorderRestart() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }

  if (pendingRecorderRestart) {
    return;
  }

  pendingRecorderRestart = true;
  isRestartingRecorder = true;
  try {
    mediaRecorder.stop();
  } catch (error) {
    console.error("Kon MediaRecorder niet stoppen voor herstart:", error);
    pendingRecorderRestart = false;
    isRestartingRecorder = false;
  }
}

async function handleDataAvailable(event) {
  if (isRestartingRecorder) {
    return;
  }

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

  const recorder = event.target || mediaRecorder;
  const rawMimeType =
    event.data.type || recorder?.mimeType || mediaRecorder?.mimeType || "";
  const detectionChunks = bufferChunks.slice();
  bufferChunks = [];
  bufferedDurationMs = 0;

  const cleanMimeType =
    (await resolveMimeType(detectionChunks, [
      rawMimeType,
      recorder?.mimeType,
      mediaRecorder?.mimeType,
      "audio/webm",
    ])) || "audio/webm";
  const blob = new Blob(detectionChunks, { type: cleanMimeType });
  const extension = mimeTypeToExtension(cleanMimeType);

  const formData = new FormData();
  formData.append("audio", blob, `spraak.${extension}`);
  formData.append("from", sourceLanguageSelect.value);
  formData.append("to", targetLanguageSelect.value);
  formData.append("textOnly", textOnlyCheckbox.checked ? "true" : "false");

  const unsupportedByDeepL = [
    "sw",
    "am",
    "mg",
    "lingala",
    "kikongo",
    "tshiluba",
    "baloué",
    "dioula",
  ];

  if (unsupportedByDeepL.includes(targetLanguageSelect.value)) {
    alert("⚠️ This language isn't supported by DeepL. AI will translate instead.");
  }

  const mimeNeedsRestart =
    cleanMimeType.includes("mp4") || extension === "mp4" || extension === "m4a";

  try {
    const response = await fetch("/api/translate", { method: "POST", body: formData });
    const data = await response.json();

    if (data.error) {
      console.error("Vertaalfout:", data.error);
    } else {
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
    }
  } catch (error) {
    console.error("Fout bij versturen van audio:", error);
  } finally {
    if (mimeNeedsRestart) {
      scheduleRecorderRestart();
    }
  }
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
      initializeMediaRecorder(stream, options);

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





