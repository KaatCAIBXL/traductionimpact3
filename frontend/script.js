document.getElementById("start").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    await setupAudioDetection(stream);

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
      audioBitsPerSecond: 128000
    });

    mediaRecorder.start(5000); // elke 5 sec automatisch fragment

    mediaRecorder.ondataavailable = async (event) => {
      const now = Date.now();
      const stilte = now - lastSpeechTime;

      if (stilte < 3000) {
        bufferChunks.push(event.data); // spreker blijft praten
      } else if (bufferChunks.length > 0) {
        const blob = new Blob(bufferChunks, { type: "audio/webm" });
        bufferChunks = [];

        const formData = new FormData();
        formData.append("audio", blob, "spraak.webm");
        formData.append("from", document.getElementById("sourceLanguage").value);
        formData.append("to", document.getElementById("languageSelect").value);
        formData.append("textOnly", document.getElementById("textOnly").checked ? "true" : "false");

        const unsupportedByDeepL = ["sw", "am", "mg", "lingala", "kikongo", "tshiluba", "baloué", "dioula"];

        if (unsupportedByDeepL.includes(document.getElementById("languageSelect").value)) {
          alert("This language isn't supported by Deepl and is translated by AI");
        }

        const response = await fetch("/api/translate", { method: "POST", body: formData });
        const data = await response.json();

        document.getElementById("transcript").innerHTML += `<p>${data.original}</p>`;
        document.getElementById("orig").innerHTML += `<p>${data.original}</p>`;
        document.getElementById("trans").innerHTML += `<p>${data.translation}</p>`;

        if (!document.getElementById("textOnly").checked) {
          spreekVertaling(data.translation, document.getElementById("languageSelect").value);
        }
      }
    };

    document.getElementById("start").disabled = true;
    document.getElementById("pause").disabled = false;
    document.getElementById("stop").disabled = false;

  } catch (err) {
    alert("Microfoon werkt niet: " + err.message);
    console.error("Microfoonfout:", err);
  }
};
