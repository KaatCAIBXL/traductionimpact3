
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
import mimetypes
import tempfile, os, threading, asyncio, textwrap
from openai import OpenAI
import deepl
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
from pydub.effects import normalize, low_pass_filter
import edge_tts



# --------------------------------this is for an app configuration and the needed apikeys
app = Flask("traductionimpact3")
CORS(app)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print(f"[!] Kon OpenAI client niet initialiseren: {exc}")

deepl_translator = None
if DEEPL_API_KEY:
    try:
        deepl_translator = deepl.Translator(DEEPL_API_KEY)
    except Exception as exc:
        print(f"[!] Kon DeepL vertaler niet initialiseren: {exc}")

vorige_zinnen = []
context_zinnen = []

ENKEL_TEKST_MODUS = False

DEFAULT_STEM = "en-US-AriaNeural"
STEMMAP = {
    "nl": "nl-NL-ColetteNeural",
    "fr": "fr-FR-DeniseNeural",
    "en": DEFAULT_STEM,
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
    "lingala": "sw-KE-ZuriNeural",
    "tshiluba": "sw-KE-ZuriNeural",
    "baloué": "sw-KE-ZuriNeural",
    "kikongo": "sw-KE-ZuriNeural",
    "malagasy": "sw-KE-ZuriNeural",
    "dioula": "sw-KE-ZuriNeural",
}


# ---------------------------------------------------------------------home page

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)




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
    Opdracht 1: bekijk alle vorige zinnen door de context te lezen. Ga dan na of er een woord is in de nieuwe zin die niet logisch is of niet in de context of zin past. Als dat het geval is vervang dan het woord door een woord dat wel in de context of zin past en dezelfde klanken heeft.  
    Opdracht 2: Als je een bijbeltekst herkent uit een erkende bijbelvertaling, zorg dat die klopt.
    Opdracht 3: Als je merkt dat er gebed is, kijk dan naar {instructies_correctie} om woorden te corrigeren. 
    Context: "{context}"
    Nieuwe zin: "{nieuwe_zin}"
    Geef enkel de verbeterde zin terug, zonder uitleg.
    """

    if openai_client is None:
        return nieuwe_zin

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

    if openai_client is None:
        return jsonify({"error": "OpenAI API-sleutel ontbreekt of client kon niet initialiseren."}), 503

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

        return jsonify({"recognized": tekst, "corrected": corrected})

    except Exception as e:
        return jsonify({"error": f"Fout bij transcriptie: {str(e)}"}), 502
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

#-------------------------------------------------------------text -> speech
# 🔊 TTS via edge_tts
def _select_stem(taalcode: str) -> str:
    return STEMMAP.get(taalcode.lower(), DEFAULT_STEM)


def _generate_tts_file(tekst: str, taalcode: str) -> str:
    stem = _select_stem(taalcode)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        pad = tmp.name
    asyncio.run(edge_tts.Communicate(tekst, stem).save(pad))
    return pad


@app.route("/api/speak", methods=["POST"])
def spreek():
    tekst = request.form.get("text")
    taalcode = request.form.get("lang", "nl")
    spreek_uit = request.form.get("speak", "true") == "true"

    if not spreek_uit or not tekst:
        return jsonify({"error": "Geen tekst om uit te spreken"}), 400

    try:
        mp3_bestand = _generate_tts_file(tekst, taalcode)

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


def spreek_tekst_synchroon(tekst: str, taalcode: str, spreek_uit: bool = True) -> None:
    if not spreek_uit or not tekst:
        return

    mp3_bestand = None
    try:
        mp3_bestand = _generate_tts_file(tekst, taalcode)
    except Exception as e:
        print(f"[!] Fout bij achtergrond-TTS: {e}")
    finally:
        if mp3_bestand and os.path.exists(mp3_bestand):
            try:
                os.remove(mp3_bestand)
            except OSError as e:
                print(f"[!] Kon tijdelijk TTS-bestand niet verwijderen: {e}")

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
    sound = AudioSegment.from_file(input_path)   # herkent mp4, webm, wav, etc.
    wav_path = input_path + ".wav"
    sound.export(wav_path, format="wav")
    return wav_path


def _determine_temp_suffix(audio_file):
    """Bepaal een veilige extensie voor het tijdelijke bestand."""

    allowed_ext = {
        ".webm",
        ".wav",
        ".mp3",
        ".mp4",
        ".m4a",
        ".ogg",
        ".opus",
        ".flac",
        ".aac",
        ".3gp",
        ".3g2",
    }

    filename = (getattr(audio_file, "filename", "") or "").split(";")[0].strip()
    _, ext = os.path.splitext(filename)
    if ext and ext.lower() in allowed_ext:
        return ext

    mimetype = (getattr(audio_file, "mimetype", "") or "").split(";")[0].strip()

    mimetype_map = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/ogg": ".ogg",
        "video/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".mp4",
        "video/mp4": ".mp4",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/x-m4a": ".m4a",
        "audio/3gpp": ".3gp",
        "audio/3gpp2": ".3g2",
        "audio/ogg; codecs=opus": ".ogg",
    }

    if mimetype in mimetype_map:
        return mimetype_map[mimetype]

    guessed = mimetypes.guess_extension(mimetype) if mimetype else None
    if guessed and guessed.lower() in allowed_ext:
        return guessed

    return ".webm"


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

    temp_input_path = None
    audio_path = None
    converted_path = None

    if openai_client is None:
        return (
            jsonify({"error": "OpenAI API-sleutel ontbreekt of client kon niet initialiseren."}),
            503,
        )

    try:
        suffix = _determine_temp_suffix(audio_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            temp_input_path = tmp.name

        audio_path = temp_input_path

        if not suffix.lower().endswith(".wav"):
            try:
                converted_path = convert_to_wav(temp_input_path)
                audio_path = converted_path
            except CouldntDecodeError as e:
                print(f"[!] Kon audio niet decoderen (ffmpeg ontbreekt?). Gebruik origineel bestand: {e}")
                audio_path = temp_input_path
            except Exception as e:
                print(f"[!] Fout bij converteren naar wav: {e}")
                return jsonify({"error": f"Kon audio niet converteren: {str(e)}"}), 400

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

        if doel_taal in deepl_supported and deepl_translator is not None:
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

            if openai_client is None:
                vertaling = verbeterde_zin
            else:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                vertaling = response.choices[0].message.content.strip()

        else:
            # Fallback naar GPT voor andere niet-DeepL talen
            prompt = f"Vertaal deze zin van {bron_taal} naar {doel_taal}: {verbeterde_zin}"
            if openai_client is None:
                vertaling = verbeterde_zin
            else:
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
        return jsonify(
            {
                "recognized": tekst,
                "corrected": verbeterde_zin,
                "translation": vertaling,
            }
        )

    except Exception as e:
        print(f"[!] Onverwachte fout: {e}")
        return jsonify({"error": str(e)}), 502

    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
#-------------------------------------------------------einde document
@app.route("/resultaat")
def resultaat():
    return send_from_directory(".", "live_vertaal.html")

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)













