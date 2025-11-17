from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
import mimetypes
import tempfile, os, threading, asyncio, textwrap, subprocess, shutil
import unicodedata
from typing import Optional
from types import SimpleNamespace
from openai import OpenAI
import deepl
from dotenv import load_dotenv
from datetime import datetime
import re

from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
from pydub.effects import normalize, low_pass_filter
import edge_tts

try:
    import whisper
except ImportError:
    whisper = None



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

LOCAL_WHISPER_MODEL_NAME = os.getenv("LOCAL_WHISPER_MODEL", "base")
_local_whisper_model = None
_local_whisper_lock = threading.Lock()

deepl_translator = None
if DEEPL_API_KEY:
    try:
        deepl_translator = deepl.Translator(DEEPL_API_KEY)
    except Exception as exc:
        print(f"[!] Kon DeepL vertaler niet initialiseren: {exc}")

vorige_zinnen = []
context_zinnen = []

ONGEWENSTE_TRANSCRIPTIES = [
    "ondertitels ingediend door de amara.org gemeenschap",
    "tv gelderland 2023",
    "bedankt om te luisteren",
]


def verwijder_ongewenste_transcripties(tekst: str) -> str:
    if not tekst:
        return tekst

    opgeschoond = tekst
    for fragment in ONGEWENSTE_TRANSCRIPTIES:
        patroon = re.compile(
            rf"\s*['\"“”‘’]*{re.escape(fragment)}['\"“”‘’]*\s*",
            flags=re.IGNORECASE,
        )
        opgeschoond = patroon.sub(" ", opgeschoond)

    return re.sub(r"\s{2,}", " ", opgeschoond).strip()

ENKEL_TEKST_MODUS = False

DEFAULT_STEM = "en-US-AriaNeural"
STEMMAP = {
    "nl": "nl-NL-ColetteNeural",
    "fr": "fr-FR-DeniseNeural",
    "en": DEFAULT_STEM,
    "es": "es-ES-ElviraNeural",
    "pt": "pt-BR-FranciscaNeural",
    "fi": "fi-FI-SelmaNeural",
    "sv": "sv-SE-SofieNeural",
}

SUPPORTED_WHISPER_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".aac",
    ".amr",
}


def _normalize_language_key(code: Optional[str]) -> str:
    if not code:
        return ""

    normalized = unicodedata.normalize("NFKD", code)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


WHISPER_LANGUAGE_OVERRIDES = {
    "lingala": "ln",
    "kituba": "kg",
    "kikongo": "kg",
    "tshiluba": "lu",
    "tshi-luba": "lu",
    "baloue": None,
    "dioula": None,
}


def map_whisper_language_hint(code: Optional[str]) -> Optional[str]:
    key = _normalize_language_key(code)
    if not key:
        return None

    if key in WHISPER_LANGUAGE_OVERRIDES:
        return WHISPER_LANGUAGE_OVERRIDES[key]

    if len(key) == 2:
        return key

    if key.startswith("en-"):
        return "en"

    return None

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
def _ensure_local_whisper_model():
    if whisper is None:
        raise RuntimeError(
            "Local Whisper fallback is niet beschikbaar: installeer het pakket 'openai-whisper'"
        )

    global _local_whisper_model
    with _local_whisper_lock:
        if _local_whisper_model is None:
            print(f"[i] Whisper API niet beschikbaar → laad lokaal model '{LOCAL_WHISPER_MODEL_NAME}'")
            _local_whisper_model = whisper.load_model(LOCAL_WHISPER_MODEL_NAME)

    return _local_whisper_model


def _transcribe_with_local_whisper(path: str, *, language_hint: Optional[str] = None):
    model = _ensure_local_whisper_model()
    options = {"fp16": False}
    if language_hint:
        options["language"] = language_hint

    result = model.transcribe(path, **options)
    text = result.get("text", "") if isinstance(result, dict) else ""
    return SimpleNamespace(text=text)


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Geen audiobestand ontvangen"}), 400

    audio_file = request.files["audio"]
    taalcode = request.form.get("lang", "fr")
    language_hint = map_whisper_language_hint(taalcode)

    if openai_client is None and whisper is None:
        return jsonify(
            {"error": "Geen OpenAI API en geen lokaal Whisper-model beschikbaar."}
        ), 503

    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            audio_path = tmp.name

        sound = AudioSegment.from_file(audio_path)
        sound = normalize(sound)
        sound = low_pass_filter(sound, cutoff=3000)
        trimmed = silence.strip_silence(sound, silence_thresh=-40)
        trimmed.export(audio_path, format="wav")

        
        transcript_response = None
        if openai_client is not None:
            try:
                with open(audio_path, "rb") as af:
                    request_kwargs = {"model": "whisper-1", "file": af}
                    if language_hint:
                        request_kwargs["language"] = language_hint

                    transcript_response = openai_client.audio.transcriptions.create(
                        **request_kwargs
                    )
            except Exception as exc:
                print(
                    "[!] Whisper API niet beschikbaar voor /api/transcribe, probeer lokaal fallback:",
                    exc,
                )

        if transcript_response is None:
            transcript_response = _transcribe_with_local_whisper(
                audio_path, language_hint=language_hint
            )

        ruwe_tekst = transcript_response.text.strip()
        tekst = verwijder_ongewenste_transcripties(ruwe_tekst)

        if tekst:
            corrected = corrigeer_zin_met_context(tekst, context_zinnen)
            corrected = verwijder_ongewenste_transcripties(corrected)
            context_zinnen.append(corrected)
        else:
            corrected = ""
        
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
def _sniff_file_extension(path: str) -> Optional[str]:
    """Lees de header van ``path`` en probeer het werkelijke formaat te bepalen.

    Safari levert bijvoorbeeld soms ``audio/mp4`` blobs aan, ook wanneer de
    frontend de extensie ``.webm`` opgeeft. Hierdoor mislukt de conversie door
    ffmpeg en krijgen we een foutmelding van het type "Invalid data found when
    processing input". Door het bestand hier opnieuw te inspecteren kunnen we
    de juiste extensie forceren voordat we Whisper of ffmpeg aanroepen.
    """

    try:
        with open(path, "rb") as handle:
            header = handle.read(16)
    except OSError:
        return None

    if len(header) < 8:
        return None

    if header.startswith(b"\x1aE\xdf\xa3"):
        return "webm"
        
    if header[4:8] == b"ftyp":
        return "mp4"
        
    if header.startswith(b"OggS"):
        return "ogg"

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"

    if header.startswith(b"#!AMR"):
        return "amr"

    return None


def convert_to_wav(input_path):
    """Converteer elk ondersteund audiobestand naar wav.

    We proberen eerst via pydub/ffmpeg. Als dat niet lukt (bijvoorbeeld omdat de
    container ffmpeg niet kan vinden of omdat een specifiek formaat niet
    ondersteund wordt), doen we een directe ffmpeg-aanroep als fallback. Wanneer
    beide strategieën falen, laten we de oorspronkelijke fout doorbubbelen zodat
    de aanroeper daar gepast op kan reageren."""

    wav_path = input_path + ".wav"
    sniffed_format = _sniff_file_extension(input_path)

    try:
        if sniffed_format:
            sound = AudioSegment.from_file(input_path, format=sniffed_format)
        else:
            sound = AudioSegment.from_file(input_path)  # herkent mp4, webm, wav, etc.
        sound.export(wav_path, format="wav")
        return wav_path
    except CouldntDecodeError:
        # Als pydub het niet redt, probeer een directe ffmpeg fallback.
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise

        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",
        ]

        if sniffed_format:
            ffmpeg_cmd.extend(["-f", sniffed_format])

        ffmpeg_cmd.extend(
            [
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ]
        )

        try:
            completed = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_output = exc.stderr.decode("utf-8", errors="ignore")
            trimmed_output = "\n".join(stderr_output.strip().splitlines()[-5:])
            raise RuntimeError(
                "FFmpeg kon het bestand niet converteren: "
                f"{trimmed_output or 'onbekende fout'}"
            ) from exc

        if completed.returncode == 0 and os.path.exists(wav_path):
            return wav_path

        raise RuntimeError("FFmpeg conversie gaf geen uitvoerbestand.")


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
        ".amr",
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
        "audio/aac": ".aac",
        "audio/aacp": ".aac",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/x-m4a": ".m4a",
        "audio/amr": ".amr",
        "audio/amr-wb": ".amr",
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


def _sniff_file_extension(path: str) -> Optional[str]:
    """Lees de header van ``path`` en probeer het werkelijke formaat te bepalen.

    Safari levert bijvoorbeeld soms ``audio/mp4`` blobs aan, ook wanneer de
    frontend de extensie ``.webm`` opgeeft. Hierdoor mislukt de conversie door
    ffmpeg en krijgen we een foutmelding van het type "Invalid data found when
    processing input". Door het bestand hier opnieuw te inspecteren kunnen we
    de juiste extensie forceren voordat we Whisper of ffmpeg aanroepen.
    """

    try:
        with open(path, "rb") as handle:
            header = handle.read(16)
    except OSError:
        return None

    if len(header) < 8:
        return None

    if header.startswith(b"\x1aE\xdf\xa3"):
        return "webm"

    if header[4:8] == b"ftyp":
        return "mp4"

    if header.startswith(b"OggS"):
        return "ogg"

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"

    return None


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
    whisper_language_hint = map_whisper_language_hint(bron_taal)

    temp_input_path = None
    audio_path = None
    converted_path = None

    if (
        doel_taal != bron_taal
        and deepl_translator is None
        and openai_client is None
    ):
        return (
            jsonify(
                {
                    "error": "Geen DeepL- of OpenAI-API geconfigureerd, vertalen is niet mogelijk.",
                    "errorCode": "missing_translation_api",
                }
            ),
            503,
        )


    

    try:
        suffix = _determine_temp_suffix(audio_file)
        suffix_lower = suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            temp_input_path = tmp.name

        audio_path = temp_input_path

        detected_extension = _sniff_file_extension(temp_input_path)
        if detected_extension:
            detected_suffix = f".{detected_extension.lower()}"
            if detected_suffix != suffix_lower:
                base, _ = os.path.splitext(temp_input_path)
                corrected_path = base + detected_suffix
                try:
                    os.replace(temp_input_path, corrected_path)
                    temp_input_path = corrected_path
                    audio_path = corrected_path
                    suffix_lower = detected_suffix
                except OSError as exc:
                    print(
                        "[!] Kon tijdelijke audio niet hernoemen naar gedetecteerde extensie:",
                        exc,
                    )

        transcript_response = None
        transcript_error = None

        def _transcribe_with_whisper(path):
            def _call_remote(include_language_hint: bool):
                if openai_client is None:
                    raise RuntimeError("Geen OpenAI-client beschikbaar")

                with open(path, "rb") as af:
                    request_kwargs = {
                        "model": "whisper-1",
                        "file": af,
                    }

                    if include_language_hint and whisper_language_hint:
                        request_kwargs["language"] = whisper_language_hint

                    return openai_client.audio.transcriptions.create(**request_kwargs)

            def _call_with_language_hint():
                if not whisper_language_hint:
                    return _call_remote(False)

                try:
                    return _call_remote(True)
                except Exception as exc:
                    lower_message = str(exc).lower()
                    if "language" not in lower_message:
                        raise

                    print(
                        "[!] Whisper-taalhint werd geweigerd, probeer automatisch detecteren:",
                        exc,
                    )
                    return _call_remote(False)

            def _call_with_fallback():
                if openai_client is None:
                    return _transcribe_with_local_whisper(
                        path, language_hint=whisper_language_hint
                    )

                try:
                    return _call_with_language_hint()
                except Exception as exc:
                    print(
                        "[!] Whisper API faalde, val terug op lokaal model indien beschikbaar:",
                        exc,
                    )
                    if whisper is None:
                        raise
                    return _transcribe_with_local_whisper(
                        path, language_hint=whisper_language_hint
                    )

            return _call_with_fallback()


        needs_conversion = suffix_lower not in SUPPORTED_WHISPER_EXTENSIONS

        if not needs_conversion:
            try:
                transcript_response = _transcribe_with_whisper(audio_path)
            except Exception as exc:
                transcript_error = exc
                needs_conversion = True
                print(
                    "[!] Directe Whisper-transcriptie mislukt, probeer conversie naar wav als fallback:",
                    exc,
                )

        if transcript_response is None and needs_conversion:
            try:
                converted_path = convert_to_wav(temp_input_path)
                audio_path = converted_path
            except CouldntDecodeError as e:
                print(
                    "[!] Kon audio niet decoderen (ffmpeg ontbreekt?). Gebruik origineel bestand:",
                    e,
                )
                audio_path = temp_input_path
            except RuntimeError as e:
                print(f"[!] Fout bij converteren naar wav (ffmpeg fallback): {e}")
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                print(f"[!] Fout bij converteren naar wav: {e}")
                return jsonify({"error": f"Kon audio niet converteren: {str(e)}"}), 400

            ext_for_whisper = os.path.splitext(audio_path)[1].lower()
            if ext_for_whisper not in SUPPORTED_WHISPER_EXTENSIONS:
                boodschap = (
                    "Dit audioformaat wordt niet ondersteund. Probeer een andere browser of "
                    "controleer of ffmpeg correct geïnstalleerd is."
                )
                print(f"[!] Niet-ondersteund formaat voor Whisper: {ext_for_whisper}")
                return jsonify({"error": boodschap}), 400

            try:
                transcript_response = _transcribe_with_whisper(audio_path)
            except Exception as exc:
                transcript_error = exc

        if transcript_response is None:
            foutmelding = "Kon audio niet transcriberen."
            if transcript_error is not None:
                foutmelding += f" (Whisper-fout: {transcript_error})"
            print(f"[!] {foutmelding}")
            return jsonify({"error": foutmelding}), 502

       ruwe_tekst = transcript_response.text.strip()
        if not ruwe_tekst:
            return jsonify({"error": "Geen spraak gedetecteerd."}), 400

        tekst = verwijder_ongewenste_transcripties(ruwe_tekst)

        # ✍️ Contextuele correctie
        if tekst:
            verbeterde_zin = corrigeer_zin_met_context(tekst, vorige_zinnen)
            verbeterde_zin = verwijder_ongewenste_transcripties(verbeterde_zin)
            vorige_zinnen.append(verbeterde_zin)
        else:
            verbeterde_zin = ""
 

        # 🌍 Vertaling
        vertaling = ""
        if verbeterde_zin:
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

























