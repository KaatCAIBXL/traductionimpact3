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
let ttsAudioElement = null;
let ttsAudioObjectUrl = null;
// Track welke zinnen al zijn voorgelezen om duplicates te voorkomen
let spokenSentences = new Map(); // Map van zin -> aantal keer voorgelezen

const micStatusElement = document.getElementById("micStatus");
const startButton = document.getElementById("start");
const pauseButton = document.getElementById("pause");
const stopButton = document.getElementById("stop");
const textOnlyCheckbox = document.getElementById("textOnly");
const microphoneSelect = document.getElementById("microphoneSelect");
const sourceLanguageSelect = document.getElementById("sourceLanguage");
const targetLanguageSelect = document.getElementById("languageSelect");
const interpreterLanguageSelect = document.getElementById("interpreterLanguage");
const transcriptContainer = document.getElementById("transcriptContainer");
let micStatusState = "idle";

const CHUNK_INTERVAL_MS = 1500;
const SILENCE_FLUSH_MS = 1400;
const MAX_BUFFER_MS = 6000;
const MIN_VALID_AUDIO_BYTES = 1024;
const MIN_UPLOAD_DURATION_MS = 1000;
const MIN_UPLOAD_BYTES = 4096;
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

  // TTS wordt al aangeroepen in queueSegmentForOutput zodra er een vertaling binnenkomt
  // Hier hoeven we alleen nog de complete/finalized versie voor te lezen als die anders is
  // Maar duplicate detection in spreekVertaling voorkomt dubbele voorlezing
  if (!textOnlyCheckbox.checked && cleaned.translation && cleaned.translation.trim()) {
    // Alleen voorlezen als dit een complete zin is en nog niet voorgelezen
    if (sentenceLooksComplete(cleaned.translation)) {
      console.log("[TTS] Finalize: controleer of complete vertaling moet worden voorgelezen:", cleaned.translation.substring(0, 50) + "...");
      // spreekVertaling heeft nu duplicate detection, dus kan veilig worden aangeroepen
      spreekVertaling(cleaned.translation, targetLanguageSelect.value);
    }
  } else {
    if (textOnlyCheckbox.checked) {
      console.log("[TTS] Overgeslagen: text-only modus actief (no voice)");
    } else if (!cleaned.translation || !cleaned.translation.trim()) {
      console.log("[TTS] Overgeslagen: geen vertaling beschikbaar");
    }
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

  // TTS direct aanroepen wanneer er een nieuwe vertaling binnenkomt
  // (ook als zin nog niet compleet is, zodat gebruiker niet hoeft te wachten)
  // Dit gebeurt vanaf het begin, tenzij "no voice" is aangevinkt
  if (!textOnlyCheckbox.checked && segment.translation && segment.translation.trim()) {
    console.log("[TTS] Nieuwe vertaling ontvangen, start voorlezen:", segment.translation.substring(0, 50) + "...");
    spreekVertaling(segment.translation, targetLanguageSelect.value);
  } else if (textOnlyCheckbox.checked) {
    console.log("[TTS] Overgeslagen: text-only modus actief (no voice)");
  }

  // Finalize alleen als zin compleet is of geforceerd moet worden
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

// ======================================================
//  MICROFOON DEVICE SELECTIE
// ======================================================
async function loadMicrophoneDevices() {
  if (!microphoneSelect) {
    return;
  }

  try {
    // Vraag eerst toestemming voor microfoon om device labels te krijgen
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach(track => track.stop()); // Stop direct, we hebben alleen toestemming nodig

    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(device => device.kind === "audioinput");

    // Leeg de dropdown behalve de standaard optie
    microphoneSelect.innerHTML = '<option value="">Standaard microfoon</option>';

    audioInputs.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      // Gebruik label of maak een beschrijvende naam
      const label = device.label || `Microfoon ${index + 1}`;
      option.textContent = label;
      microphoneSelect.appendChild(option);
    });

    console.log(`[Microfoon] ${audioInputs.length} microfoon(s) gevonden`);
  } catch (error) {
    console.warn("[Microfoon] Kon microfoons niet ophalen:", error);
    // Voeg een optie toe om later opnieuw te proberen
    if (microphoneSelect.options.length === 1) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Klik 'start' om microfoons te laden";
      microphoneSelect.appendChild(option);
    }
  }
}

// Laad microfoons bij het laden van de pagina
if (microphoneSelect) {
  loadMicrophoneDevices();
}

function renderLatestSegments() {
  if (!transcriptContainer) {
    return;
  }

  // Bouw HTML voor alle segmenten (we verwijderen GEEN oude items meer,
  // zodat het downloadbestand altijd de volledige sessie bevat).
  let html = "";
  for (let i = 0; i < sessionSegments.length; i++) {
    const segment = sessionSegments[i];
    const hasContent = (segment.recognized && segment.recognized.trim()) ||
                      (segment.corrected && segment.corrected.trim()) ||
                      (segment.translation && segment.translation.trim());
    
    if (!hasContent) {
      continue;
    }
    
    html += `<div class="segment" style="margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #e0e0e0;">`;
    
    if (segment.recognized && segment.recognized.trim()) {
      html += `<div style="margin-bottom: 8px;"><strong>Transcriptie:</strong> <span>${escapeHtml(segment.recognized)}</span></div>`;
    }
    
    if (segment.corrected && segment.corrected.trim()) {
      html += `<div style="margin-bottom: 8px;"><strong>Correctie:</strong> <span>${escapeHtml(segment.corrected)}</span></div>`;
    }
    
    if (segment.translation && segment.translation.trim()) {
      html += `<div style="margin-bottom: 8px;"><strong>Vertaling:</strong> <span>${escapeHtml(segment.translation)}</span></div>`;
    }
    
    html += `</div>`;
  }
  
  transcriptContainer.innerHTML = html;

  // Altijd automatisch naar beneden scrollen zodat de nieuwste tekst zichtbaar is,
  // maar we houden ALLE vorige segmenten in het geheugen voor de download.
  setTimeout(() => {
    const container = transcriptContainer;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  }, 10);
}

renderLatestSegments();
function stopActiveStream() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }
}

function triggerSilenceFlush() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }

  if (pendingSilenceFlush) {
    return;
  }

  pendingSilenceFlush = true;

  try {
    mediaRecorder.requestData();
  } catch (error) {
    pendingSilenceFlush = false;
    console.warn("Kon recorderflush niet aanvragen:", error);
  }
}

function releaseAudioResources() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }

  stopRecorderDataPump();

  pendingSilenceFlush = false;
  resetPendingSentence();


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

  const silenceDuration = Date.now() - lastSpeechTime;
  if (!isSpeaking && silenceDuration >= SILENCE_FLUSH_MS) {
    triggerSilenceFlush();
  }

    previousSpeakingState = isSpeaking;
  }, 150);
}


// ======================================================
//  TTS VIA BACKEND
// ======================================================
function ensureTtsAudioElement() {
  if (ttsAudioElement && document.body.contains(ttsAudioElement)) {
    return ttsAudioElement;
  }

  ttsAudioElement = document.createElement("audio");
  ttsAudioElement.id = "ttsPlayer";
  ttsAudioElement.preload = "auto";
  ttsAudioElement.style.display = "none";
  document.body.appendChild(ttsAudioElement);
  return ttsAudioElement;
}

// Functie om TTS audio te stoppen
function stopTtsAudio() {
  if (ttsAudioElement && !ttsAudioElement.paused) {
    ttsAudioElement.pause();
    ttsAudioElement.currentTime = 0;
    console.log("[TTS] Audio gestopt");
  }
}

function playTtsBlob(blob, lang) {
  const audioEl = ensureTtsAudioElement();

  // Stop huidige audio als die speelt
  if (!audioEl.paused) {
    audioEl.pause();
    audioEl.currentTime = 0;
  }

  // Cleanup oude URL
  if (ttsAudioObjectUrl) {
    URL.revokeObjectURL(ttsAudioObjectUrl);
    ttsAudioObjectUrl = null;
  }

  const nextUrl = URL.createObjectURL(blob);
  ttsAudioObjectUrl = nextUrl;

  // Reset audio element
  audioEl.src = "";
  audioEl.load(); // Force reload

  // Set nieuwe source
  audioEl.src = nextUrl;
  audioEl.dataset.lang = lang || "";
  audioEl.currentTime = 0;
  audioEl.volume = 1.0; // Zorg dat volume op max staat

  // Add event listeners for debugging (alleen eenmalig)
  if (!audioEl.hasAttribute("data-listeners-added")) {
    audioEl.setAttribute("data-listeners-added", "true");
    audioEl.onplay = () => console.log("[TTS] ✅ Audio start afspelen");
    audioEl.onended = () => console.log("[TTS] ✅ Audio afgelopen");
    audioEl.onerror = (e) => {
      console.error("[TTS] ❌ Audio fout:", e, audioEl.error);
    };
    audioEl.onloadstart = () => console.log("[TTS] 📥 Audio begint te laden");
    audioEl.oncanplay = () => console.log("[TTS] ✅ Audio kan afgespeeld worden");
    audioEl.oncanplaythrough = () => console.log("[TTS] ✅ Audio volledig geladen");
  }

  // Probeer af te spelen
  const playPromise = audioEl.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise
      .then(() => {
        console.log("[TTS] ✅ Audio afspelen succesvol gestart");
      })
      .catch((error) => {
        console.error("[TTS] ❌ Kon audio niet afspelen:", error.name, error.message);
        // Browser autoplay policy might block this - user interaction required
        if (error.name === "NotAllowedError") {
          console.warn("[TTS] ⚠️ Browser blokkeert autoplay - klik ergens op de pagina en probeer opnieuw");
          // Probeer opnieuw na korte delay (soms helpt dit)
          setTimeout(() => {
            audioEl.play().catch(e => console.error("[TTS] ❌ Retry ook mislukt:", e));
          }, 100);
        } else {
          console.error("[TTS] ❌ Onbekende audio fout:", error);
        }
      });
  } else {
    // Fallback voor oudere browsers
    console.log("[TTS] ⚠️ play() geeft geen promise terug, probeer direct af te spelen");
    try {
      audioEl.play();
    } catch (e) {
      console.error("[TTS] ❌ Direct play() mislukt:", e);
    }
  }
}

async function spreekVertaling(text, lang) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    console.log("[TTS] Geen tekst om voor te lezen");
    return;
  }

  // Duplicate detection: voorkom dat dezelfde zin meer dan 2 keer wordt voorgelezen
  const normalizedText = trimmed.toLowerCase().trim();
  const count = spokenSentences.get(normalizedText) || 0;
  
  if (count >= 2) {
    console.log(`[TTS] ⚠️ Zin al ${count} keer voorgelezen, overslaan: "${trimmed.substring(0, 50)}..."`);
    return;
  }
  
  // Verhoog teller
  spokenSentences.set(normalizedText, count + 1);
  console.log(`[TTS] Zin wordt voorgelezen (${count + 1}/2): "${trimmed.substring(0, 50)}${trimmed.length > 50 ? '...' : ''}"`);

  console.log(`[TTS] Vraag audio aan voor: "${trimmed.substring(0, 50)}${trimmed.length > 50 ? '...' : ''}" (taal: ${lang})`);

  const formData = new FormData();
  formData.append("text", trimmed);
  formData.append("lang", lang);
  formData.append("speak", "true");

  let response;
  try {
    response = await fetch("/api/speak", { method: "POST", body: formData });
  } catch (networkError) {
    console.error("[TTS] Netwerkfout tijdens ophalen van spraak:", networkError);
    return;
  }

  if (!response.ok) {
    let errorMessage = "Kon geen audio genereren";
    try {
      const payload = await response.json();
      if (payload && typeof payload.error === "string" && payload.error.trim()) {
        errorMessage = payload.error.trim();
      }
    } catch (_) {
      // Het antwoord was geen JSON; val terug op standaardmelding.
    }

    console.error(`[TTS] Serverfout (${response.status}): ${errorMessage}`);
    return;
  }

  try {
    const blob = await response.blob();
    if (!blob || blob.size === 0) {
      console.error("[TTS] Lege audio blob ontvangen");
      return;
    }
    console.log(`[TTS] Audio ontvangen (${blob.size} bytes), start afspelen...`);
    playTtsBlob(blob, lang);
  } catch (blobError) {
    console.error("[TTS] Kon audiobestand niet verwerken:", blobError);
  }
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


  const forceFlush = pendingSilenceFlush;
  if (pendingSilenceFlush) {
    pendingSilenceFlush = false;
  }

  const shouldFlush =
    bufferChunks.length > 0 &&
    (forceFlush || stilte >= SILENCE_FLUSH_MS || bufferedDurationMs >= MAX_BUFFER_MS);

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

  if (totalBytes < MIN_UPLOAD_BYTES || bufferedDurationMs < MIN_UPLOAD_DURATION_MS) {
    if (forceFlush) {
      pendingSilenceFlush = true;
    }
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

  const flushedDueToSilence = forceFlush;

  const formData = new FormData();
  formData.append("audio", blob, `spraak.${extension}`);
  formData.append("from", sourceLanguageSelect.value);
  formData.append("to", targetLanguageSelect.value);
  formData.append("textOnly", textOnlyCheckbox.checked ? "true" : "false");
  if (interpreterLanguageSelect && interpreterLanguageSelect.value) {
    formData.append("interpreter_lang", interpreterLanguageSelect.value);
  }

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
      // Debug: log wat we ontvangen
      console.log("[Translate] Response ontvangen:", {
        recognized: data.recognized?.substring(0, 50),
        corrected: data.corrected?.substring(0, 50),
        translation: data.translation?.substring(0, 50),
        hasTranslation: !!(data.translation && data.translation.trim())
      });

      const segment = {
        recognized: data.recognized || "",
        corrected: data.corrected || data.recognized || "",
        translation: data.translation || "",
        silenceDetected: Boolean(data.silenceDetected),
        forceFinalize: flushedDueToSilence,
      };

      queueSegmentForOutput(segment);
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
    resetPendingSentence();
    sessionSegments = [];
    spokenSentences.clear(); // Reset duplicate tracking bij nieuwe sessie
    renderLatestSegments();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const message = "Browser ondersteunt geen microfoon opname.";
      alert("❌ " + message);
      setMicStatus("error", message);
      return;
    }

    try {
      setMicStatus("calibrating");
      
      // Gebruik de geselecteerde microfoon deviceId
      const constraints = { audio: true };
      const selectedDeviceId = microphoneSelect?.value;
      if (selectedDeviceId) {
        constraints.audio = { deviceId: { exact: selectedDeviceId } };
        console.log(`[Microfoon] Gebruik geselecteerde microfoon: ${microphoneSelect.options[microphoneSelect.selectedIndex]?.textContent}`);
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      activeStream = stream;

      console.log("🎙️ Microfoon toestemming OK");
      
      // Herlaad microfoons om labels te krijgen (na toestemming zijn ze beschikbaar)
      await loadMicrophoneDevices();
      
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
    if (!mediaRecorder) {
      console.warn("[Pause] Geen mediaRecorder beschikbaar");
      return;
    }

    const state = mediaRecorder.state;
    console.log(`[Pause] Huidige state: ${state}, isPaused: ${isPaused}`);

    if (state === "recording" && !isPaused) {
      // Pauzeren: stop zowel opname als TTS
      try {
        triggerSilenceFlush();
        mediaRecorder.pause();
        stopTtsAudio(); // Stop TTS audio
        isPaused = true;
        pauseButton.innerText = "▶️ continue ▶️";
        setMicStatus("paused", "Opname gepauzeerd");
        console.log("[Pause] Opname en TTS gepauzeerd");
      } catch (error) {
        console.error("[Pause] Fout bij pauzeren:", error);
      }
    } else if (state === "paused" && isPaused) {
      // Hervatten
      try {
        mediaRecorder.resume();
        isPaused = false;
        pauseButton.innerText = "⏸️ pause ⏸️";
        setMicStatus("listening", "Aan het luisteren...");
        console.log("[Pause] Opname hervat");
      } catch (error) {
        console.error("[Pause] Fout bij hervatten:", error);
      }
    } else {
      console.warn(`[Pause] Onverwachte state: ${state}, isPaused: ${isPaused}`);
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
    finalizePendingSentence(true);
    downloadSessionDocument();
    setMicStatus("idle");
  };
}

function downloadSessionDocument() {
  if (!sessionSegments.length) {
    console.warn("[Download] Geen segmenten om te downloaden");
    alert("Geen transcriptie om te downloaden.");
    return;
  }

  try {
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
    
    if (!content || content.trim().length === 0) {
      console.warn("[Download] Lege inhoud om te downloaden");
      alert("Geen inhoud om te downloaden.");
      return;
    }

    // Validate content encoding
    let blob;
    try {
      blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    } catch (blobError) {
      console.error("[Download] Fout bij maken van blob:", blobError);
      // Fallback: probeer zonder charset
      try {
        blob = new Blob([content], { type: "text/plain" });
      } catch (fallbackError) {
        console.error("[Download] Fallback blob creatie faalde:", fallbackError);
        alert("Erreur lors du téléchargement du fichier: Kon bestand niet maken.");
        return;
      }
    }
    
    if (!blob || blob.size === 0) {
      console.error("[Download] Kon blob niet maken of blob is leeg");
      alert("Erreur lors du téléchargement du fichier: Kon bestand niet maken.");
      return;
    }

    // Sanitize filename - remove invalid characters for all operating systems
    // Invalid characters: < > : " / \ | ? * and control characters (0-31)
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    // Remove any remaining invalid characters
    const sanitizedTimestamp = timestamp.replace(/[<>:"/\\|?*\x00-\x1F]/g, "");
    const filename = `transcriptie-${sanitizedTimestamp}.txt`;

    // Try method 1: Create download link
    let url;
    let link;
    try {
      url = URL.createObjectURL(blob);
      if (!url) {
        throw new Error("URL.createObjectURL returned null");
      }

      link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.style.display = "none";
      link.setAttribute("download", filename); // Extra attribute for better browser support
      
      document.body.appendChild(link);
      
      // Trigger download with error handling
      try {
        link.click();
        console.log(`[Download] Bestand gedownload: ${filename} (${blob.size} bytes)`);
      } catch (clickError) {
        console.error("[Download] Fout bij link.click():", clickError);
        // Fallback: open in new window
        window.open(url, "_blank");
        alert("Download gestart. Als de download niet automatisch start, controleer uw browser instellingen.");
      }
      
      // Cleanup
      setTimeout(() => {
        try {
          if (link && link.parentNode) {
            document.body.removeChild(link);
          }
          if (url) {
            URL.revokeObjectURL(url);
          }
        } catch (cleanupError) {
          console.warn("[Download] Fout bij cleanup:", cleanupError);
        }
      }, 1000);
      
    } catch (urlError) {
      console.error("[Download] Fout bij maken van download URL:", urlError);
      
      // Fallback method 2: Data URL (for smaller files)
      if (blob.size < 2 * 1024 * 1024) { // Only for files < 2MB
        try {
          const reader = new FileReader();
          reader.onload = function(e) {
            const dataUrl = e.target.result;
            const fallbackLink = document.createElement("a");
            fallbackLink.href = dataUrl;
            fallbackLink.download = filename;
            fallbackLink.style.display = "none";
            document.body.appendChild(fallbackLink);
            fallbackLink.click();
            setTimeout(() => {
              document.body.removeChild(fallbackLink);
            }, 1000);
            console.log(`[Download] Bestand gedownload via fallback: ${filename}`);
          };
          reader.onerror = function() {
            alert("Erreur lors du téléchargement du fichier: Kon bestand niet lezen.");
          };
          reader.readAsDataURL(blob);
        } catch (fallbackError) {
          console.error("[Download] Fallback methode faalde:", fallbackError);
          alert("Erreur lors du téléchargement du fichier: " + (fallbackError.message || "Onbekende fout"));
        }
      } else {
        alert("Erreur lors du téléchargement du fichier: Bestand te groot voor alternatieve methode.");
      }
    }
  } catch (error) {
    console.error("[Download] Fout bij downloaden:", error);
    alert("Erreur lors du téléchargement du fichier: " + (error.message || "Onbekende fout"));
  }
}

















