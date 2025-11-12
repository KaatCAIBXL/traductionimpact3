// Even checken of het script echt geladen is
console.log("✅ script.js loaded");

let mediaRecorder;
let bufferChunks = [];
let isSpeaking = false;

let audioContext;
let analyser;
let source;
let lastSpeechTime = Date.now();

let isPaused = false;
let intervalId = null;

// --------------------------
// VOLUME / SPREKEN-DETECTIE
// --------------------------
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
    if (isSpeaking) lastSpeechTime = Date.now();
  }, 200);
}

// --------------------------
// TTS: vertaling laten spreken
// --------------------------
async function spreekVertaling(text, lang) {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("lang", lang);
  formData.append("speak", "true");

  const response = await fetch("/api/speak", {
    method: "POST",
    body: formData
  });

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}

// --------------------------
// START KNOP
// --------------------------
document.getElementById("start").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log("🎙️ Microfoon toegestaan");
    await setupAudioDetection(stream);

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
      audioBitsPerSecond: 128000
    });

    mediaRecorder.start(5000); // elke 5 sec fragment

    mediaRecorder.ondataavailable = async (event) => {
      const now = Date.now();
      const stilte = now - lastSpeechTime;

      if (stilte < 3000) {
        // spreker nog bezig
        bufferChunks.push(event.data);
      } else if (bufferChunks.length > 0) {
        // spreker klaar: versturen
        const blob = new Blob(bufferChunks, { type: "audio/webm" });
        bufferChunks = [];

        const formData = new FormData();
        formData.append("audio", blob, "spraak.webm");
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
          alert("This language isn't supported by Deepl and is translated by AI");
        }

        const response = await fetch("/api/translate", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        // tekst op scherm
        document.getElementById("transcript").innerHTML += `<p>${data.original}</p>`;
        document.getElementById("orig").innerHTML       += `<p>${data.original}</p>`;
        document.getElementById("trans").innerHTML      += `<p>${data.translation}</p>`;

        // laten uitspreken (indien niet text-only)
        if (!document.getElementById("textOnly").checked) {
          spreekVertaling(
            data.translation,
            document.getElementById("languageSelect").value
          );
        }
      }
    };

    // knoppen-status
    document.getElementById("start").disabled = true;
    document.getElementById("pause").disabled = false;
    document.getElementById("stop").disabled  = false;

  } catch (err) {
    alert("Microfoon werkt niet: " + err.message);
    console.error("Microfoonfout:", err);
  }
};

// --------------------------
// PAUZE KNOP
// --------------------------
document.getElementById("pause").onclick = () => {
  if (!mediaRecorder) return;

  if (!isPaused && mediaRecorder.state === "recording") {
    mediaRecorder.pause();
    isPaused = true;
    document.getElementById("pause").innerText = "▶️ continue ▶️";
  } else if (isPaused && mediaRecorder.state === "paused") {
    mediaRecorder.resume();
    isPaused = false;
    document.getElementById("pause").innerText = "⏸️ pause ⏸️";
  }
};

// --------------------------
// STOP KNOP
// --------------------------
document.getElementById("stop").onclick = () => {
  clearInterval(intervalId);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  document.getElementById("start").disabled = false;
  document.getElementById("pause").disabled = true;
  document.getElementById("stop").disabled  = true;
  document.getElementById("pause").innerText = "⏸️ pause ⏸️";
  isPaused = false;
};
// Even checken of het script echt geladen is
console.log("✅ script.js loaded");

let mediaRecorder;
let bufferChunks = [];
let isSpeaking = false;

let audioContext;
let analyser;
let source;
let lastSpeechTime = Date.now();

let isPaused = false;
let intervalId = null;

// --------------------------
// VOLUME / SPREKEN-DETECTIE
// --------------------------
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
    if (isSpeaking) lastSpeechTime = Date.now();
  }, 200);
}

// --------------------------
// TTS: vertaling laten spreken
// --------------------------
async function spreekVertaling(text, lang) {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("lang", lang);
  formData.append("speak", "true");

  const response = await fetch("/api/speak", {
    method: "POST",
    body: formData
  });

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}

// --------------------------
// START KNOP
// --------------------------
document.getElementById("start").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log("🎙️ Microfoon toegestaan");
    await setupAudioDetection(stream);

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
      audioBitsPerSecond: 128000
    });

    mediaRecorder.start(5000); // elke 5 sec fragment

    mediaRecorder.ondataavailable = async (event) => {
      const now = Date.now();
      const stilte = now - lastSpeechTime;

      if (stilte < 3000) {
        // spreker nog bezig
        bufferChunks.push(event.data);
      } else if (bufferChunks.length > 0) {
        // spreker klaar: versturen
        const blob = new Blob(bufferChunks, { type: "audio/webm" });
        bufferChunks = [];

        const formData = new FormData();
        formData.append("audio", blob, "spraak.webm");
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
          alert("This language isn't supported by Deepl and is translated by AI");
        }

        const response = await fetch("/api/translate", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        // tekst op scherm
        document.getElementById("transcript").innerHTML += `<p>${data.original}</p>`;
        document.getElementById("orig").innerHTML       += `<p>${data.original}</p>`;
        document.getElementById("trans").innerHTML      += `<p>${data.translation}</p>`;

        // laten uitspreken (indien niet text-only)
        if (!document.getElementById("textOnly").checked) {
          spreekVertaling(
            data.translation,
            document.getElementById("languageSelect").value
          );
        }
      }
    };

    // knoppen-status
    document.getElementById("start").disabled = true;
    document.getElementById("pause").disabled = false;
    document.getElementById("stop").disabled  = false;

  } catch (err) {
    alert("Microfoon werkt niet: " + err.message);
    console.error("Microfoonfout:", err);
  }
};

// --------------------------
// PAUZE KNOP
// --------------------------
document.getElementById("pause").onclick = () => {
  if (!mediaRecorder) return;

  if (!isPaused && mediaRecorder.state === "recording") {
    mediaRecorder.pause();
    isPaused = true;
    document.getElementById("pause").innerText = "▶️ continue ▶️";
  } else if (isPaused && mediaRecorder.state === "paused") {
    mediaRecorder.resume();
    isPaused = false;
    document.getElementById("pause").innerText = "⏸️ pause ⏸️";
  }
};

// --------------------------
// STOP KNOP
// --------------------------
document.getElementById("stop").onclick = () => {
  clearInterval(intervalId);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  document.getElementById("start").disabled = false;
  document.getElementById("pause").disabled = true;
  document.getElementById("stop").disabled  = true;
  document.getElementById("pause").innerText = "⏸️ pause ⏸️";
  isPaused = false;
};
