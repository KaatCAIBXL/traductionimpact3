
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

let audioContext;
let analyser;
let source;
let lastSpeechTime = Date.now();

let isPaused = false;
let intervalId = null;


// ======================================================
//  SPREKEN-DETECTIE
// ======================================================
async function setupAudioDetection(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  intervalId = setInterval(() => {
    analyser.getByteFrequencyData(dataArray);
    const volume = dataArray.reduce((a, b) => a + b) / dataArray.length;
    isSpeaking = volume > 10;

    if (isSpeaking) {
      lastSpeechTime = Date.now();
    }
  }, 200);
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
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    console.log("🎙️ Microfoon toestemming OK");
    await setupAudioDetection(stream);

    const options = getRecorderOptions();
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.start(5000); // elke 5 sec fragment

    mediaRecorder.ondataavailable = async (event) => {
      const now = Date.now();
      const stilte = now - lastSpeechTime;

      if (stilte < 3000) {
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

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  document.getElementById("start").disabled = false;
  document.getElementById("pause").disabled = true;
  document.getElementById("stop").disabled = true;
  document.getElementById("pause").innerText = "⏸️ pause ⏸️ ";

  isPaused = false;
};
