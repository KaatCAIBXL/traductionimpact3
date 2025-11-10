from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
import tempfile, os, threading, asyncio, textwrap
from openai import OpenAI
import deepl
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

from pydub import AudioSegment, silence
from pydub.effects import normalize, low_pass_filter
import edge_tts


# --------------------------------this is for an app configuration and the needed apikeys
app = Flask("traductionimpact3")
CORS(app)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepl_translator = deepl.Translator(DEEPL_API_KEY)

vorige_zinnen = []
context_zinnen = []

ENKEL_TEKST_MODUS = False


# ---------------------------------------------------------------------home page
@app.route("/")
def index():
    return send_from_directory("index.html")

#--------------------------------------------------------------------corrected sentences
def corrigeer_zin_met_context(nieuwe_zin, vorige_zinnen):
    if not nieuwe_zin.strip():
        return nieuwe_zin

    context = " ".join(vorige_zinnen[-3:])
    try:
        with open("instructies_correctie.txt", "r", encoding="utf-8") as f:
            instructies_correctie = f.read()
    except FileNotFoundError:
        instructies_correctie = "(Geen instructies gevonden.)"

    prompt = f"""
    Opdracht 1: bekijk alle vorige zinnen door de context te lezen. Ga dan na of er een woord is in de nieuwe zin die niet in de context past. 
    Opdracht 2: Als je een bijbeltekst herkent uit een erkende bijbelvertaling, zorg dat die klopt.
    Opdracht 3: Als je merkt dat er gebed is, kijk dan naar {instructies_correctie}.
    Context: "{context}"
    Nieuwe zin: "{nieuwe_zin}"
    Geef enkel de verbeterde zin terug, zonder uitleg.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Fout bij contextuele correctie: {e}")
        return nieuwe_zin


#-------------------------------------------------------------------------correcting with whisper
@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Geen audiobestand ontvangen"}), 400

    audio_file = request.files["audio"]
    taalcode = request.form.get("lang", "fr")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            audio_path = tmp.name

        sound = AudioSegment.from_file(audio_path)
        sound = normalize(sound)
        sound = low_pass_filter(sound, cutoff=3000)
        trimmed = silence.strip_silence(sound, silence_thresh=-40)
        trimmed.export(audio_path, format="wav")

        with open(audio_path, "rb") as af:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1", file=af, language=taalcode
            )

        tekst = transcript_response.text.strip()
        corrected = corrigeer_zin_met_context(tekst, context_zinnen)
        context_zinnen.append(corrected)

        return jsonify({"original": tekst, "corrected": corrected})

    except Exception as e:
        return jsonify({"error": f"Fout bij transcriptie: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

#-------------------------------------------------------------text -> speech
# 🔊 TTS via edge_tts
@app.route("/api/speak", methods=["POST"])
def spreek():
    tekst = request.form.get("text")
    taalcode = request.form.get("lang", "nl")
    spreek_uit = request.form.get("speak", "true") == "true"

    if not spreek_uit or not tekst:
        return jsonify({"error": "Geen tekst om uit te spreken"}), 400

    stemmap = {
        "nl": "nl-NL-ColetteNeural",
        "fr": "fr-FR-DeniseNeural",
        "en": "en-US-AriaNeural",
        "de": "de-DE-KatjaNeural",
        "es": "es-ES-ElviraNeural",
        "pt": "pt-BR-FranciscaNeural",
        "fi": "fi-FI-SelmaNeural",
        "sv": "sv-SE-SofieNeural",
        "no": "nb-NO-PernilleNeural",
        "pl": "pl-PL-AgnieszkaNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "tr": "tr-TR-EmelNeural",
        "ja": "ja-JP-NanamiNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ar": "ar-EG-SalmaNeural",
        "hi": "hi-IN-SwaraNeural",
        "id": "id-ID-GadisNeural",
        "ms": "ms-MY-YasminNeural",
        "sw": "sw-KE-ZuriNeural",
        "am": "am-ET-MekdesNeural",
        "lingala": "sw-KE-ZuriNeural",  # vervangstem
        "tshiluba": "sw-KE-ZuriNeural",  # vervangstem
        "baloué": "sw-KE-ZuriNeural",  # vervansgtem
        "kikongo": "sw-KE-ZuriNeural",  # vervansgtem
        "malagasy": "sw-KE-ZuriNeural",  # vervansgtem
        "dioula": "sw-KE-ZuriNeural",  # vervansgtem
    }

    stem = stemmap.get(taalcode.lower(), "en-US-AriaNeural")
    mp3_bestand = "tts_audio.mp3"

    try:
        asyncio.run(edge_tts.Communicate(tekst, stem).save(mp3_bestand))

        @after_this_request
        def remove_file(response):
            try:
                os.remove(mp3_bestand)
            except Exception as e:
                print(f"[!] Kon bestand niet verwijderen: {e}")
            return response

        return send_file(mp3_bestand, mimetype="audio/mpeg")

    except Exception as e:
        return jsonify({"error": f"Fout bij stemgeneratie: {str(e)}"}), 500

# 🔁 DeepL taalcode mapping
def map_vertaling_taalcode_deepl(taalcode):
    code = taalcode.lower()
    if code in ["en", "en-us"]:
        return "EN-US"
    elif code in ["pt", "pt-br"]:
        return "PT-BR"
    elif code in ["zh", "zh-cn", "zh-hans"]:
        return "ZH"
    else:
        return code.upper()

#----------------------------------------------------audiobestand omvormen

def convert_to_wav(input_path):
    sound = AudioSegment.from_file(input_path)
    wav_path = input_path.replace(".webm", ".wav")
    sound.export(wav_path, format="wav")
    return wav_path


# -------------------- HOOFDROUTE --------------------
@app.route("/api/translate", methods=["POST"])
def vertaal_audio():
    global vorige_zinnen

    if "audio" not in request.files:
        return jsonify({"error": "Geen audio ontvangen"}), 400

    audio_file = request.files["audio"]
    bron_taal = request.form.get("from", "fr").lower()
    doel_taal = request.form.get("to", "nl").lower()
    enkel_tekst = request.form.get("textOnly", "false") == "true"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        audio_path = convert_to_wav(audio_path)

    try:
        # 🎧 Transcriptie via Whisper
        with open(audio_path, "rb") as af:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1", file=af, language=bron_taal
            )
        tekst = transcript_response.text.strip()
        if not tekst:
            return jsonify({"error": "Geen spraak gedetecteerd."}), 400

        # ✍️ Contextuele correctie
        verbeterde_zin = corrigeer_zin_met_context(tekst, vorige_zinnen)
        vorige_zinnen.append(verbeterde_zin)

        # 🌍 Vertaling
        deepl_supported = {
            "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "id",
            "it", "ja", "ko", "lt", "lv", "nb", "nl", "pl", "pt", "ro", "ru", "sk",
            "sl", "sv", "tr", "uk", "zh"
        }

        if doel_taal in deepl_supported:
            try:
                doel_taal_code = map_vertaling_taalcode_deepl(doel_taal)
                result = deepl_translator.translate_text(
                    verbeterde_zin, source_lang=bron_taal, target_lang=doel_taal_code
                )
                vertaling = result.text
            except Exception:
                vertaling = verbeterde_zin  # fallback bij DeepL-fout

        elif doel_taal == "tshiluba":
            try:
                with open("instructies_Tshiluba.txt", "r", encoding="utf-8") as f:
                    insTsh = f.read()
            except FileNotFoundError:
                insTsh = "(Geen instructies gevonden.)"

            prompt = textwrap.dedent(f"""
                Vertaal deze zin van {bron_taal} naar Tshiluba: {verbeterde_zin}
                MAAR als er een woord is dat je niet kent, kijk dan naar deze lijst: {insTsh}
                Als je het woord nog steeds niet kent, kies een gelijkaardig woord met vergelijkbare betekenis.
                Als dat ook niet lukt, vertaal dan naar het Frans als fallback.
            """)

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            vertaling = response.choices[0].message.content.strip()

        else:
            # Fallback naar GPT voor andere niet-DeepL talen
            prompt = f"Vertaal deze zin van {bron_taal} naar {doel_taal}: {verbeterde_zin}"
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            vertaling = response.choices[0].message.content.strip()

        # 🔊 Spraakuitvoer (indien niet in tekst-only modus)
        threading.Thread(
            target=spreek_tekst_synchroon,
            args=(vertaling, doel_taal, not enkel_tekst),
        ).start()

        # 🔁 Antwoord naar frontend
        return jsonify({"original": verbeterde_zin, "translation": vertaling})

    except Exception as e:
        print(f"[!] Onverwachte fout: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
#-------------------------------------------------------einde document
@app.route("/resultaat")
def resultaat():
    return send_from_directory(".", "live_vertaal.html")

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



