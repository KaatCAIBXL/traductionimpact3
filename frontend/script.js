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
let cachedMp4InitSegment = null;
let cachedWebmHeader = null;
let recorderRequestTimer = null;
let pendingSilenceFlush = false;
let pendingSentence = null;

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
const MIN_VALID_AUDIO_BYTES = 1024;
// Zorg dat elke blob die we naar de backend sturen opnieuw een container-header bevat
// (Safari/Chrome leveren anders "headerloze" segmenten waardoor Whisper niets kan).
const FORCE_RECORDER_RESTART_AFTER_UPLOAD = true;
const MAX_INIT_SEGMENT_BYTES = 128 * 1024;
let sessionSegments = [];

function resetPendingSentence() {
  pendingSentence = null;
}

function textJoin(left = "", right = "") {
  const a = (left || "").trimEnd();
  const b = (right || "").trimStart();

  if (!a) return b;
  if (!b) return a;

  const needsSpace =
    !/[\s\-–—(\[]$/.test(a) && !/^[,.;:!?…)]/.test(b);

  return needsSpace ? `${a} ${b}` : `${a}${b}`;
}

function sentenceLooksComplete(text = "") {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }

  return /[.!?…](?:['")\]]*|\s*)$/.test(trimmed);
}

function finalizePendingSentence(force = false) {
  if (!pendingSentence) {
    return;
  }

  const cleaned = {
    recognized: (pendingSentence.recognized || "").trim(),
    corrected: (pendingSentence.corrected || "").trim(),
    translation: (pendingSentence.translation || "").trim(),
  };

  if (!cleaned.recognized && !cleaned.corrected && !cleaned.translation) {
    pendingSentence = null;
    return;
  }

  if (!force && !sentenceLooksComplete(cleaned.corrected) && !sentenceLooksComplete(cleaned.translation)) {
    return;
  }

  pendingSentence = null;
  sessionSegments.push(cleaned);
  renderLatestSegments();

  if (!textOnlyCheckbox.checked && cleaned.translation) {
    spreekVertaling(cleaned.translation, targetLanguageSelect.value);
  }
}

function queueSegmentForOutput(segment) {
  const hasContent =
    (segment.recognized && segment.recognized.trim()) ||
    (segment.corrected && segment.corrected.trim()) ||
    (segment.translation && segment.translation.trim());

  if (!hasContent) {
    if (segment.silenceDetected) {
      finalizePendingSentence(true);
    }
    return;
  }

  if (!pendingSentence) {
    pendingSentence = { ...segment };
  } else {
    pendingSentence.recognized = textJoin(pendingSentence.recognized, segment.recognized);
    pendingSentence.corrected = textJoin(pendingSentence.corrected, segment.corrected);
    pendingSentence.translation = textJoin(pendingSentence.translation, segment.translation);
  }

  if (
    sentenceLooksComplete(pendingSentence.corrected) ||
    sentenceLooksComplete(pendingSentence.translation) ||
    segment.forceFinalize
  ) {
    const force = Boolean(segment.forceFinalize);
    finalizePendingSentence(force);
  }
}

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

  stopRecorderDataPump();

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
  cachedMp4InitSegment = null;
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

    // Maak de drempel adaptief zodat we ook zachte stemmen detecteren.
    const adaptiveBump = noiseFloorRms + 0.0015;
    const multiplicativeBump = noiseFloorRms * 2.2;
    const silenceThreshold = Math.max(0.0025, adaptiveBump, multiplicativeBump);
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
  const preferredTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4;codecs=mp4a.40.2",
    "audio/mp4",
    "audio/3gpp",
    "audio/aac",
    "audio/wav",
  ];

  for (const candidate of preferredTypes) {
    if (MediaRecorder.isTypeSupported(candidate)) {
      if (candidate !== "audio/webm" && candidate !== "audio/webm;codecs=opus") {
        console.warn(`⚠️ Schakel MediaRecorder over naar ${candidate} (webm niet ondersteund)`);
      }
      return { mimeType: candidate };
    }
  }

  console.warn("⚠️ Geen bekende mimeType ondersteund — laat browser standaard kiezen");
  return {};
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

async function extractMp4InitSegment(blob) {
  try {
    const buffer = await blob.arrayBuffer();
    const view = new DataView(buffer);
    let offset = 0;
    let endOfHeader = 0;

    while (offset + 8 <= view.byteLength) {
      const atomSize = view.getUint32(offset);
      if (atomSize < 8) {
        break;
      }

      const atomType = String.fromCharCode(
        view.getUint8(offset + 4),
        view.getUint8(offset + 5),
        view.getUint8(offset + 6),
        view.getUint8(offset + 7)
      );

      if (["ftyp", "moov", "free", "skip"].includes(atomType)) {
        endOfHeader = offset + atomSize;
        offset += atomSize;
        if (atomType === "moov") {
          break;
        }
        continue;
      }

      if (atomType === "mdat" || atomType === "moof") {
        break;
      }

      offset += atomSize;
    }

    if (endOfHeader > 0) {
      const headerSize = Math.min(endOfHeader, MAX_INIT_SEGMENT_BYTES);
      return buffer.slice(0, headerSize);
    }
  } catch (error) {
    console.warn("Kon mp4-initsegment niet extraheren:", error);
  }

  return null;
}
function findByteSequence(haystack, needle) {
  if (!haystack || !needle || !needle.length || haystack.length < needle.length) {
    return -1;
  }

  outer: for (let i = 0; i <= haystack.length - needle.length; i += 1) {
    for (let j = 0; j < needle.length; j += 1) {
      if (haystack[i + j] !== needle[j]) {
        continue outer;
      }
    }
    return i;
  }

  return -1;
}

function extractWebmHeaderBytes(uint8Array) {
  if (!uint8Array || uint8Array.length < 4) {
    return null;
  }

  const CLUSTER_ID = [0x1f, 0x43, 0xb6, 0x75];
  const clusterIndex = findByteSequence(uint8Array, CLUSTER_ID);

  if (clusterIndex > 0) {
    const sliceEnd = Math.min(clusterIndex, MAX_INIT_SEGMENT_BYTES);
    return uint8Array.slice(0, sliceEnd).buffer;
  }

  const fallbackEnd = Math.min(uint8Array.length, MAX_INIT_SEGMENT_BYTES);
  return uint8Array.slice(0, fallbackEnd).buffer;
}

async function ensureChunkHasContainerHeader(chunk, rawMimeType = "") {
  if (!chunk || !chunk.size) {
    return chunk;
  }

  const cleanMimeType = sanitizeMimeType(rawMimeType || chunk.type || "");

  if (cleanMimeType === "audio/mp4") {
    const sniffed = await sniffMimeTypeFromChunks([chunk]);

    if (sniffed === "audio/mp4") {
      const initSegment = await extractMp4InitSegment(chunk);
      if (initSegment) {
        cachedMp4InitSegment = initSegment;
      }
      return chunk;
    }

    if (!sniffed && cachedMp4InitSegment) {
      try {
        const chunkBuffer = await chunk.arrayBuffer();
        const combined = new Uint8Array(
          cachedMp4InitSegment.byteLength + chunkBuffer.byteLength
        );
        combined.set(new Uint8Array(cachedMp4InitSegment), 0);
        combined.set(new Uint8Array(chunkBuffer), cachedMp4InitSegment.byteLength);
        return new Blob([combined], { type: "audio/mp4" });
      } catch (error) {
        console.warn("Kon mp4-fragment niet samenvoegen met init-segment:", error);

      }
    }
  }

  if (cleanMimeType === "audio/webm" || cleanMimeType === "video/webm") {
    try {
      const buffer = await chunk.arrayBuffer();
      const uint8 = new Uint8Array(buffer);

      const hasEbmlHeader =
        uint8.length >= 4 &&
        uint8[0] === 0x1a &&
        uint8[1] === 0x45 &&
        uint8[2] === 0xdf &&
        uint8[3] === 0xa3;

      if (hasEbmlHeader) {
        const headerBytes = extractWebmHeaderBytes(uint8);
        if (headerBytes) {
          cachedWebmHeader = headerBytes;
        }
        return chunk;
      }

      if (cachedWebmHeader) {
        const cachedHeaderArray = new Uint8Array(cachedWebmHeader);
        const combined = new Uint8Array(
          cachedHeaderArray.byteLength + uint8.byteLength
        );
        combined.set(cachedHeaderArray, 0);
        combined.set(uint8, cachedHeaderArray.byteLength);
        return new Blob([combined], { type: cleanMimeType });
      }
    } catch (error) {
      console.warn("Kon webm-header niet reconstrueren:", error);
    }
  }

  return chunk;
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
  ensureRecorderDataPump(recorder);
  mediaRecorder = recorder;
  return recorder;
}

function stopRecorderDataPump() {
  if (recorderRequestTimer) {
    clearInterval(recorderRequestTimer);
    recorderRequestTimer = null;
  }
}

function ensureRecorderDataPump(recorder) {
  stopRecorderDataPump();
  if (!recorder) {
    return;
  }

  recorderRequestTimer = setInterval(() => {
    if (!recorder || recorder.state !== "recording") {
      return;
    }

    if (isPaused) {
      return;
    }

    try {
      recorder.requestData();
    } catch (error) {
      console.warn("Kon recordergegevens niet opvragen:", error);
      stopRecorderDataPump();
    }
  }, CHUNK_INTERVAL_MS);
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
  const chunkMimeType =
    (event.data && event.data.type) ||
    mediaRecorder?.mimeType ||
    recorderOptions?.mimeType ||
    "";

  if (event.data && event.data.size) {
    const chunkWithHeader = await ensureChunkHasContainerHeader(
      event.data,
      chunkMimeType
    );
    bufferChunks.push(chunkWithHeader);
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
    chunkMimeType || recorder?.mimeType || mediaRecorder?.mimeType || "";
  const detectionChunks = bufferChunks.slice();
  const totalBytes = detectionChunks.reduce(
    (sum, chunk) => sum + (chunk?.size || 0),
    0
  );

  if (totalBytes < MIN_VALID_AUDIO_BYTES) {
    // Safari levert soms losse containerheaders zonder audiogegevens aan. Die
    // veroorzaken "File ended prematurely"-fouten bij ffmpeg. Wacht tot er
    // effectieve audio binnenloopt zodat we de header samen met echte data
    // kunnen versturen.
    return;
  }

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
    FORCE_RECORDER_RESTART_AFTER_UPLOAD ||
    cleanMimeType.includes("mp4") ||
    extension === "mp4" ||
    extension === "m4a";

  try {
    const response = await fetch("/api/translate", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok || data.error) {
      const foutmelding = data?.error || `Serverfout (${response.status})`;
      console.error("Vertaalfout:", foutmelding);
      setMicStatus("error", foutmelding);
      if (data?.errorCode === "missing_translation_api") {
        alert(
          "❌ Geen vertaal-API's geconfigureerd. Vul een DEEPL_API_KEY of OPENAI_API_KEY in op de server."
        );
      } else {
        alert("❌ Vertaalfout: " + foutmelding);
      }
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















