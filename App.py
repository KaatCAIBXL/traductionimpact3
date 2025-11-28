from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
import mimetypes
import tempfile, os, threading, asyncio, textwrap, subprocess, shutil
import unicodedata
from dataclasses import asdict, dataclass
from typing import Optional
from types import SimpleNamespace
from openai import OpenAI
import deepl
from dotenv import load_dotenv
from datetime import datetime
import re
import requests
import base64

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.exceptions import CouldntDecodeError
from pydub.effects import normalize, low_pass_filter
import edge_tts
from edge_tts import exceptions as edge_tts_exceptions
import numpy as np

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
MODELLAB_API_KEY = os.getenv("MODELLAB_API_KEY") or os.getenv("FLUX2PRO_API_KEY")

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

# ModelsLab Flux 2 Pro API voor beeldgeneratie
modelslab_api_key = MODELLAB_API_KEY
if modelslab_api_key:
    print("[i] ModelsLab API key geladen voor Flux 2 Pro Text To Image")
else:
    print("[!] ModelsLab API key niet gevonden. Beeldgeneratie zal niet beschikbaar zijn.")

vorige_zinnen = []
context_zinnen = []

# Zinnen die we liever niet in de uiteindelijke transcriptie tonen.
ONGEWENSTE_TRANSCRIPTIES = [
    "ondertitels ingediend door de amara.org gemeenschap",
    "Ondertitels ingediend door de amara.org gemeenschap",
    "Ondertitels ingediend door de amara.org gemeenschap",
    "Sous-titres soumis par la communauté amara.org.",
    "Sous-titres soumis par la communauté amara.org",
    "Sous titres soumis par la communaute amara.org",
    "Sous-titres réalisés par la communauté d'Amara.org",
    "Sous titres realises par la communaute d'Amara.org",
    "Sous-titres réalisés para la communauté d'Amara.org",
    "Sous titres realises para la communaute d'Amara.org",
    "Sous-titres réalisés para la communauté d'Amara.org",
    "Sous titres realises para la communaute d'Amara.org",
    "Sous-titres",
    "sous-titres",
    "Sous titres",
    "sous titres",
    "ondertiteld",
    "Ondertiteld",
    "ondertiteling",
    "Ondertiteling",
    "ondertitels",
    "Ondertitels",
    "Sous-titrage ST",
    "sous-titrage ST",
    "Sous titrage ST",
    "Ondertiteling ST",
    "ondertiteling ST",
    "501",
    "Merci. Au revoir.",
    "Merci. Au revoir",
    "Merci.",
    "Merci",
    "Ciao !",
    "Ciao!",
    "Ciao",
    "Merci d'avoir regardé cette vidéo !",
    "Merci d'avoir regarde cette video !",
    "Merci d'avoir regardé cette vidéo",
    "Merci d'avoir regarde cette video",
    "Sous-titrage Société Radio-canada",
    "Sous titrage Societe Radio-canada",
    "Sous-titrage Société Radio Canada",
    "Ondertiteling Radio-Canada",
    "ondertiteling Radio-Canada",
    "Ondertiteling Radio Canada",
    "Bedankt. Tot ziens.",
    "Bedankt. Tot ziens",
    "Tot ziens.",
    "Tot ziens",
    "Tot Ziens.",
    "Tot Ziens",
    "A bientôt",
    "A bientot",
    "À bientôt",
    "À bientot",
    "a bientôt",
    "a bientot",
    "Bedankt.",
    "Bedankt",
    "Dag!",
    "Doei.",
    "Doei",
    "doei.",
    "doei",
    "Au revoir.",
    "Au revoir",
    "au revoir.",
    "au revoir",
    "d'avoir regardé cette vidéo !",
    "d'avoir regarde cette video !",
    "d'avoir regardé cette vidéo",
    "d'avoir regarde cette video",
    "voor het kijken !",
    "Voor het kijken !",
    "voor het kijken",
    "Voor het kijken",
    "Doeg !",
    "Doeg!",
    "Doeg",
    "doeg !",
    "doeg!",
    "doeg",
    "Dag",
    "Bedankt voor het kijken naar deze video !",
    "Bedankt voor het kijken naar deze video",
    "Bedankt voor het kijken",
    "TV GELDERLAND 2021",
    "TV GELDERLAND 2023",
    "bedankt om te luisteren",
    "bedankt om te kijken",
    "bedankt om te kijken naar deze video",
    "Dankjewel en tot de volgende keer!",
    "Dankjewel en tot de volgende keer",
    "dankjewel en tot de volgende keer!",
    "dankjewel en tot de volgende keer",
    "Merci et à la prochaine fois !",
    "Merci et à la prochaine fois",
    "merci et à la prochaine fois !",
    "merci et à la prochaine fois",
    "Merci et à la prochaine fois.",
    "merci et à la prochaine fois.",
    "Dank u wel.",
    "Dank u wel",
    "dank u wel.",
    "dank u wel",
    "Dank u",
    "dank u",
    "Merci beaucoup",
    "merci beaucoup",
    "Merci beaucoup.",
    "merci beaucoup.",
    "Je vous remercie",
    "je vous remercie",
    "Je vous remercie.",
    "je vous remercie.",
    "Je te remercie",
    "je te remercie",
    "Je te remercie.",
    "je te remercie.",
]


def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _normalize_blacklist_text(value: str) -> str:
    stripped = _strip_diacritics(value.lower())
    stripped = re.sub(r"[^a-z0-9]+", " ", stripped)
    return stripped.strip()


NORMALIZED_BLACKLIST = [
    fragment_norm
    for fragment_norm in (_normalize_blacklist_text(fragment) for fragment in ONGEWENSTE_TRANSCRIPTIES)
    if fragment_norm
]

BLACKLIST_TOKEN_COMBOS = [
    ("sous titres", "amara"),
    ("sous titres", "communaut"),
    ("subtitles", "amara"),
    ("subtitles", "community"),
    ("subtitulos", "amara"),
    ("subtitulos", "comunidad"),
    ("ondertitels", "amara"),
    ("merci", "regarde"),
    ("merci", "video"),
    ("bedankt", "kijken"),
    ("bedankt", "video"),
    ("sous titrage", "radio"),
    ("sous titrage", "canada"),
    ("ondertiteling", "radio"),
    ("ondertiteling", "canada"),
]


def _contains_emoji(tekst: str) -> bool:
    """Check if text contains any emoji characters."""
    # Emoji ranges in Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(tekst))


def _has_meaningful_transcript_content(tekst: str) -> bool:
    """Return ``True`` if ``tekst`` contains non-punctuation characters."""

    for char in tekst:
        if unicodedata.category(char)[0] in {"L", "N"}:
            return True
    return False


def verwijder_ongewenste_transcripties(tekst: str) -> str:
    if not tekst:
        return tekst

    # EARLY REJECTION: Check for emojis - reject anything with emojis
    if _contains_emoji(tekst):
        return ""

    # EARLY REJECTION: Check for subtitle variants BEFORE any processing
    # This ensures they never appear in the transcript at all
    tekst_lower = tekst.lower()
    # French variants
    if ("sous" in tekst_lower and "titres" in tekst_lower) or "sous-titres" in tekst_lower or "sous titres" in tekst_lower:
        return ""
    # Dutch variants
    if "ondertiteld" in tekst_lower or "ondertiteling" in tekst_lower or ("ondertitels" in tekst_lower and "amara" in tekst_lower):
        return ""
    
    # Check for common closing phrases
    tekst_stripped = tekst_lower.strip()
    # French closing phrases
    if tekst_stripped in ["merci.", "merci", "merci. au revoir.", "merci. au revoir", "ciao !", "ciao!", "ciao", "a bientôt", "a bientot", "à bientôt", "à bientot", "merci et à la prochaine fois !", "merci et à la prochaine fois", "merci et à la prochaine fois.", "merci beaucoup", "merci beaucoup.", "je vous remercie", "je vous remercie.", "je te remercie", "je te remercie."]:
        return ""
    # Dutch closing phrases
    if tekst_stripped in ["bedankt.", "bedankt", "bedankt. tot ziens.", "bedankt. tot ziens", "tot ziens.", "tot ziens", "dag!", "dag", "dankjewel en tot de volgende keer!", "dankjewel en tot de volgende keer", "dank u wel.", "dank u wel", "dank u"]:
        return ""
    # Check for "501" (common subtitle error code)
    if tekst_stripped == "501":
        return ""
    # Check for "Sous-titrage ST" or "Ondertiteling ST"
    if "sous-titrage st" in tekst_lower or "ondertiteling st" in tekst_lower:
        return ""
    
    # Check for "Ja, ik weet het." or "Oui, je sais" variants
    if tekst_stripped in ["ja, ik weet het.", "ja, ik weet het", "oui, je sais.", "oui, je sais", "ja ik weet het.", "ja ik weet het"]:
        return ""
    
    # Check for "Ja, ik ben er" variants (common repetition issue)
    if tekst_stripped in ["ja, ik ben er.", "ja, ik ben er", "ja ik ben er.", "ja ik ben er", "ja ik ben er!", "ja, ik ben er!"]:
        return ""
    
    # Check for single-word sentences (only one word, possibly with punctuation)
    # Remove punctuation and check if only one word remains
    tekst_zonder_punctuatie = re.sub(r'[^\w\s]', '', tekst_stripped).strip()
    woorden = tekst_zonder_punctuatie.split()
    if len(woorden) == 1:
        # Single word detected - filter it out
        return ""

    opgeschoond = tekst
    for fragment in ONGEWENSTE_TRANSCRIPTIES:
        patroon = re.compile(
            rf"\s*['\"“”‘’]*{re.escape(fragment)}['\"“”‘’]*\s*",
            flags=re.IGNORECASE,
        )
        opgeschoond = patroon.sub(" ", opgeschoond)

    opgeschoond = re.sub(r"\s{2,}", " ", opgeschoond).strip()
    if not _has_meaningful_transcript_content(opgeschoond):
        return ""

    normalized_content = _normalize_blacklist_text(opgeschoond)
    for fragment_norm in NORMALIZED_BLACKLIST:
        if fragment_norm and fragment_norm in normalized_content:
            return ""

    for needle_a, needle_b in BLACKLIST_TOKEN_COMBOS:
        if needle_a in normalized_content and needle_b in normalized_content:
            return ""

    # Aggressive check: if "sous" + "titres" appear together in any form, reject
    if "sous" in normalized_content and "titres" in normalized_content:
        return ""

    return opgeschoond

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


def _to_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_int(value: Optional[str], default: int) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class AudioPreprocessingConfig:
    """Configuration options for preparing raw microphone uploads."""

    normalize_audio: bool = True
    low_pass_cutoff: Optional[int] = 3000
    silence_threshold_dbfs: float = -40.0
    min_silence_duration_ms: int = 300
    silence_padding_ms: int = 120

    @classmethod
    def from_request(cls, form_data) -> "AudioPreprocessingConfig":
        normalize_value = form_data.get("normalize")
        if normalize_value is None:
            normalize_value = form_data.get("normalizeAudio")
        cutoff_value = form_data.get("lowPassCutoff")
        cutoff = _to_int(cutoff_value, 3000) if cutoff_value not in (None, "") else 3000
        if cutoff is not None and cutoff <= 0:
            cutoff = None
        return cls(
            normalize_audio=_to_bool(normalize_value, True),
            low_pass_cutoff=cutoff,
            silence_threshold_dbfs=_to_float(form_data.get("silenceThreshold"), -40.0),
            min_silence_duration_ms=_to_int(form_data.get("minSilenceMs"), 300),
            silence_padding_ms=_to_int(form_data.get("silencePaddingMs"), 120),
        )


def _trim_silence(segment: AudioSegment, config: AudioPreprocessingConfig) -> AudioSegment:
    """Return ``segment`` without leading/trailing silence based on config settings.
    
    Uses more conservative trimming at the end to preserve speech that might be
    quieter or have trailing silence, improving transcription quality.
    """

    nonsilent_ranges = detect_nonsilent(
        segment,
        min_silence_len=max(config.min_silence_duration_ms, 1),
        silence_thresh=config.silence_threshold_dbfs,
    )
    if not nonsilent_ranges:
        return AudioSegment.silent(duration=0)

    # Use standard padding at the start
    start = max(nonsilent_ranges[0][0] - config.silence_padding_ms, 0)
    
    # Use more generous padding at the end to preserve trailing speech
    # This helps with transcription quality at the end of segments
    end_padding = config.silence_padding_ms * 2  # Double padding at the end
    end = min(nonsilent_ranges[-1][1] + end_padding, len(segment))
    
    if start >= end:
        return AudioSegment.silent(duration=0)
    
    # Ensure minimum segment length to help Whisper with transcription
    MIN_SEGMENT_LENGTH_MS = 500  # Minimum 500ms for better transcription
    trimmed = segment[start:end]
    if len(trimmed) < MIN_SEGMENT_LENGTH_MS and len(segment) >= MIN_SEGMENT_LENGTH_MS:
        # If trimmed segment is too short but original has enough content,
        # use a more conservative trim (keep more of the original)
        center = (start + end) // 2
        half_min = MIN_SEGMENT_LENGTH_MS // 2
        start = max(0, center - half_min)
        end = min(len(segment), center + half_min)
        trimmed = segment[start:end]
    
    return trimmed


def _preprocess_audio_file(path: str, config: AudioPreprocessingConfig) -> bool:
    """Apply normalization, filtering and silence trimming to ``path``.

    Returns ``True`` when speech remains after trimming, otherwise ``False``.
    """

    sound = AudioSegment.from_file(path)
    if config.normalize_audio:
        sound = normalize(sound)
    if config.low_pass_cutoff:
        sound = low_pass_filter(sound, cutoff=config.low_pass_cutoff)

    trimmed = _trim_silence(sound, config)
    if len(trimmed) == 0:
        return False

    trimmed.export(path, format="wav")
    return True


DEFAULT_GPT_TRANSLATION_MODEL = "gpt-4"
GPT_TRANSLATION_MODEL_OVERRIDES = {
    "kituba": "gpt-4.1",
    "lingala": "gpt-4.1",
    "tshiluba": "gpt-4.1",
    "malagasy": "gpt-4.1",
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


def select_gpt_translation_model(target_language: Optional[str]) -> str:
    key = _normalize_language_key(target_language)
    if key in GPT_TRANSLATION_MODEL_OVERRIDES:
        return GPT_TRANSLATION_MODEL_OVERRIDES[key]
    return DEFAULT_GPT_TRANSLATION_MODEL

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

BELANGRIJK: Als de originele zin een correcte zin is, mag je die niet veranderen.
ENKEL als je merkt dat iets onlogisch is, incorrect is, of als je een Bijbelvers tegenkomt, mag je aanpassingen doen.

KRITIEK: HERHAAL NOOIT tekst uit de context! De context is alleen bedoeld om te begrijpen wat er gezegd wordt, NIET om tekst uit de context opnieuw te gebruiken of toe te voegen aan de nieuwe zin.

Opdracht 1: Als je een Bijbeltekst uit een erkende vertaling herkent, herstel die nauwkeurig.

Opdracht 2: Als de zin een gebed bevat, pas de regels toe uit:
{instructies_correctie}

Opdracht 3: Als je een zin tegenkomt met "Ondertitels ..." of "...bedankt om te ..." in eender welke taal,
vervang dit door een lege string "". Met andere woorden: dit moet weg.

Opdracht 4: Als je een '.' tegenkomt, laat die staan. Voeg nooit extra zinnen toe!

Opdracht 5: CRITIEK - Onlogische zinnen en speech-to-text fouten:
- Als de zin HEEL onlogisch is en duidelijk niet past in de context, dan heeft speech-to-text het waarschijnlijk verkeerd verstaan.
- Als je onlogische woorden tegenkomt die niet in de context passen, probeer deze te vervangen door woorden met dezelfde klanken die WEL in de context passen.
- Gebruik de context ALLEEN om te begrijpen wat er bedoeld werd, NIET om tekst uit de context te kopiëren of herhalen.
- Pas dit toe op ALLE onlogische woorden.
- Als je echt niet kunt raden wat er bedoeld werd en de zin is compleet onlogisch, geef dan een lege string "" terug (wis de zin volledig).
- HERHAAL NOOIT woorden of zinnen die al in de context staan, ook niet als je denkt dat het logisch is.

Geef alleen de gecorrigeerde zin terug die natuurlijk klinkt, zonder uitleg. Gebruik ALLEEN de woorden uit de nieuwe zin, niet uit de context.

Geef NOOIT opmerkingen. Enkel vertaling of niets.
Context: "{context}"
Nieuwe zin: "{nieuwe_zin}"
"""

    if openai_client is None:
        print("[!] Geen OpenAI-client beschikbaar.")
        return nieuwe_zin

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
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


def _build_initial_prompt(context_list: list, max_length: int = 200) -> Optional[str]:
    """Build an initial prompt from context sentences to help Whisper with transcription."""
    if not context_list:
        return None
    
    # Use the last few sentences as context (most relevant)
    recent_context = context_list[-3:] if len(context_list) >= 3 else context_list
    prompt = " ".join(recent_context).strip()
    
    if len(prompt) > max_length:
        # Take the end of the prompt (most recent context)
        prompt = prompt[-max_length:]
    
    return prompt if prompt else None


def _transcribe_with_local_whisper(path: str, *, language_hint: Optional[str] = None, initial_prompt: Optional[str] = None):
    model = _ensure_local_whisper_model()
    options = {
        "fp16": False,
        "temperature": 0.0,  # More deterministic, better for consistent quality
        "beam_size": 5,  # Better quality for shorter segments
        "best_of": 5,  # Try multiple decodings and pick best
        "compression_ratio_threshold": 2.4,  # Filter out repetitive text
        "logprob_threshold": -1.0,  # Filter low-confidence transcriptions
        "no_speech_threshold": 0.6,  # Better detection of speech vs silence
    }
    if language_hint:
        options["language"] = language_hint
    if initial_prompt:
        # Use context from previous transcriptions to help Whisper
        options["initial_prompt"] = initial_prompt[:200]  # Limit length
    
    result = model.transcribe(path, **options)
    if isinstance(result, dict):
        text = result.get("text", "")
        language = result.get("language")
        return SimpleNamespace(text=text, language=language)
    else:
        text = result.text if hasattr(result, "text") else ""
        language = result.language if hasattr(result, "language") else None
        return SimpleNamespace(text=text, language=language)


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

        preprocess_config = AudioPreprocessingConfig.from_request(request.form)
        if not _preprocess_audio_file(audio_path, preprocess_config):
            return jsonify({
                "recognized": "",
                "corrected": "",
                "silenceDetected": True,
                "preprocessing": asdict(preprocess_config),
            })



        # Build initial prompt from context to help Whisper
        initial_prompt = _build_initial_prompt(context_zinnen)
        
        transcript_response = None
        if openai_client is not None:
            try:
                with open(audio_path, "rb") as af:
                    request_kwargs = {"model": "whisper-1", "file": af}
                    if language_hint:
                        request_kwargs["language"] = language_hint
                    if initial_prompt:
                        request_kwargs["prompt"] = initial_prompt[:200]

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
                audio_path, language_hint=language_hint, initial_prompt=initial_prompt
            )

        ruwe_tekst = transcript_response.text.strip()
        tekst = verwijder_ongewenste_transcripties(ruwe_tekst)

        if not tekst:
            return jsonify(
                {
                    "recognized": "",
                    "corrected": "",
                    "silenceDetected": True,
                    "preprocessing": asdict(preprocess_config),
                }
            )

        corrected = corrigeer_zin_met_context(tekst, context_zinnen)
        corrected = verwijder_ongewenste_transcripties(corrected)
        context_zinnen.append(corrected)

        return jsonify(
            {
                "recognized": tekst,
                "corrected": corrected,
                "preprocessing": asdict(preprocess_config),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Fout bij transcriptie: {str(e)}"}), 502
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

#-------------------------------------------------------------text -> speech
# 🔊 TTS via edge_tts
def _select_stem(taalcode: str) -> str:
    return STEMMAP.get(taalcode.lower(), DEFAULT_STEM)


def _sanitize_tts_text(tekst: Optional[str]) -> str:
    """Maak de tekst veilig voor TTS en verwijder betekenisloze whitespace."""

    if tekst is None:
        return ""
    return unicodedata.normalize("NFC", tekst).strip()


def _generate_tts_file(tekst: str, taalcode: str) -> str:
    """Maak een tijdelijk mp3-bestand met uitgesproken ``tekst``."""

    opgeschoonde_tekst = _sanitize_tts_text(tekst)
    if not opgeschoonde_tekst:
        raise ValueError("Geen tekst beschikbaar om uit te spreken.")

    stem = _select_stem(taalcode)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        pad = tmp.name

    try:
        asyncio.run(edge_tts.Communicate(opgeschoonde_tekst, stem).save(pad))
    except edge_tts_exceptions.NoAudioReceived as exc:
        if os.path.exists(pad):
            os.remove(pad)
        raise ValueError(
            "TTS-service gaf geen audio terug. Controleer of de tekst niet leeg is."
        ) from exc
    except Exception:
        if os.path.exists(pad):
            os.remove(pad)
        raise

    return pad

@app.route("/api/speak", methods=["POST"])
def spreek():
    tekst = request.form.get("text")
    taalcode = request.form.get("lang", "nl")
    spreek_uit = request.form.get("speak", "true") == "true"

    tekst = _sanitize_tts_text(tekst)
    taalcode = (taalcode or "nl").strip() or "nl"

    if not spreek_uit:
        return jsonify({"error": "Spraakuitvoer is uitgeschakeld"}), 400

    if not tekst:
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

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Fout bij stemgeneratie: {str(e)}"}), 500


def spreek_tekst_synchroon(tekst: str, taalcode: str, spreek_uit: bool = True) -> None:
    tekst = _sanitize_tts_text(tekst)

    if not spreek_uit or not tekst:
        return

    mp3_bestand = None
    try:
        mp3_bestand = _generate_tts_file(tekst, taalcode)
    except ValueError as e:
        print(f"[i] TTS overgeslagen: {e}")
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
    interpreter_lang = request.form.get("interpreter_lang", "").lower()
    interpreter_lang_hint = map_whisper_language_hint(interpreter_lang) if interpreter_lang else None
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
            def _is_audio_too_short_error(error: Exception) -> bool:
                message = str(error).lower()
                return "audio file is too short" in message or "audio_too_short" in message

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
                    
                    # Use context from previous transcriptions to help Whisper
                    initial_prompt = _build_initial_prompt(vorige_zinnen)
                    if initial_prompt:
                        request_kwargs["prompt"] = initial_prompt[:200]

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
                # Build initial prompt from context
                initial_prompt = _build_initial_prompt(vorige_zinnen)
                
                if openai_client is None:
                    return _transcribe_with_local_whisper(
                        path, language_hint=whisper_language_hint, initial_prompt=initial_prompt
                    )

                try:
                    return _call_with_language_hint()
                except Exception as exc:
                    if _is_audio_too_short_error(exc):
                        print(
                            "[i] Whisper API overslaan: audio korter dan 0.1s; behandel als stilte."
                        )
                        return SimpleNamespace(text="")

                    print(
                        "[!] Whisper API faalde, val terug op lokaal model indien beschikbaar:",
                        exc,
                    )
                    if whisper is None:
                        raise
                    return _transcribe_with_local_whisper(
                        path, language_hint=whisper_language_hint, initial_prompt=initial_prompt
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
            return jsonify(
                {
                    "recognized": "",
                    "corrected": "",
                    "translation": "",
                    "silenceDetected": True,
                }
            )

        # Check if detected language matches interpreter language - if so, ignore this transcription
        if interpreter_lang_hint:
            detected_language = None
            # Try to get language from response (OpenAI API returns it as attribute)
            if hasattr(transcript_response, "language"):
                detected_language = transcript_response.language
            elif isinstance(transcript_response, dict):
                detected_language = transcript_response.get("language")
            
            # Normalize detected language for comparison
            if detected_language:
                detected_lang_normalized = detected_language.lower().strip()
                interpreter_lang_normalized = interpreter_lang_hint.lower().strip()
                
                # Check if detected language matches interpreter language
                if detected_lang_normalized == interpreter_lang_normalized:
                    print(f"[i] Gedetecteerde taal ({detected_language}) komt overeen met tolk-taal ({interpreter_lang}), transcriptie genegeerd.")
                    return jsonify(
                        {
                            "recognized": "",
                            "corrected": "",
                            "translation": "",
                            "silenceDetected": True,
                            "interpreterFiltered": True,
                        }
                    )

        tekst = verwijder_ongewenste_transcripties(ruwe_tekst)

        if not tekst:
            return jsonify(
                {
                    "recognized": "",
                    "corrected": "",
                    "translation": "",
                    "silenceDetected": True,
                }
            )


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
                    gpt_model = select_gpt_translation_model(doel_taal)
                    response = openai_client.chat.completions.create(
                        model=gpt_model,
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
                    gpt_model = select_gpt_translation_model(doel_taal)
                    response = openai_client.chat.completions.create(
                        model=gpt_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    vertaling = response.choices[0].message.content.strip()

        else:
            # Er is geen tekst om te vertalen (bv. stilte of ruis); houd het leeg zodat
            # we geen "oei, je hebt niets doorgegeven"-berichten van GPT terugkrijgen.
            vertaling = ""

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
@app.route("/api/generate-song", methods=["POST"])
def generate_song():
    """
    Genereer een liedje uit lyrics, vergelijkbaar met Musicful.ai.
    Accepteert lyrics en genereert vocals (TTS) + optioneel instrumentale muziek.
    """
    try:
        lyrics = request.form.get("lyrics", "").strip()
        language = request.form.get("language", "fr").strip()
        genre = request.form.get("genre", "pop").strip()  # pop, rock, rap, etc.
        style = request.form.get("style", "").strip()  # Optionele stijl beschrijving
        add_instrumental = request.form.get("add_instrumental", "true").lower() == "true"
        
        if not lyrics:
            return jsonify({"error": "Geen lyrics opgegeven"}), 400
        
        # Genereer vocals met TTS
        print(f"[Song Generation] Genereer vocals voor {len(lyrics)} karakters in taal {language}")
        vocals_path = _generate_tts_file(lyrics, language)
        
        vocals_audio = AudioSegment.from_mp3(vocals_path)
        vocals_duration = len(vocals_audio)
        
        # Optioneel: voeg instrumentale muziek toe
        final_audio = vocals_audio
        instrumental_info = {"added": False}
        
        if add_instrumental:
            try:
                # Voor nu: genereer een eenvoudige instrumentale track
                # Later kan dit uitgebreid worden met MusicGen, Suno API, etc.
                instrumental_audio = _generate_simple_instrumental(
                    duration_ms=vocals_duration,
                    genre=genre,
                    style=style
                )
                
                if instrumental_audio:
                    # Mix vocals en instrumentale muziek (vocals iets luider)
                    vocals_audio = vocals_audio.apply_gain(3)  # Vocals 3dB luider
                    final_audio = vocals_audio.overlay(instrumental_audio)
                    instrumental_info = {"added": True, "genre": genre}
                    print(f"[Song Generation] Instrumentale muziek toegevoegd ({genre})")
            except Exception as e:
                print(f"[Song Generation] Kon instrumentale muziek niet toevoegen: {e}")
                # Gebruik alleen vocals als fallback
                final_audio = vocals_audio
        
        # Sla het finale liedje op
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path.close()
        final_audio.export(output_path.name, format="mp3", bitrate="192k")
        
        @after_this_request
        def cleanup_files(response):
            try:
                if os.path.exists(vocals_path):
                    os.remove(vocals_path)
                if os.path.exists(output_path.name):
                    os.remove(output_path.name)
            except Exception as e:
                print(f"[!] Kon bestanden niet verwijderen: {e}")
            return response
        
        return send_file(
            output_path.name,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"song_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        )
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[!] Fout bij song generatie: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Fout bij song generatie: {str(e)}"}), 500


def _generate_simple_instrumental(duration_ms: int, genre: str = "pop", style: str = "") -> Optional[AudioSegment]:
    """
    Genereer een eenvoudige instrumentale track.
    Dit is een basis implementatie - kan later uitgebreid worden met:
    - MusicGen (Meta)
    - Suno AI API
    - Mubert API
    - Of andere muziek generatie tools
    """
    try:
        # Voor nu: genereer een eenvoudige beat/ritme met pydub
        # Dit is een placeholder - echte implementatie vereist een muziek generatie model
        
        # Maak een eenvoudige drum beat als placeholder
        sample_rate = 44100
        duration_seconds = duration_ms / 1000.0
        
        # Genereer een eenvoudige baslijn (sine wave)
        # Eenvoudige beat pattern
        beat_freq = 60  # BPM
        beat_interval = 60.0 / beat_freq
        
        # Maak een lege audio track
        silence = AudioSegment.silent(duration=duration_ms)
        
        # Voeg een eenvoudige bas toe (placeholder)
        # In productie zou je hier een echte muziek generatie API gebruiken
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        
        # Eenvoudige baslijn (C major chord)
        bass_freq = 65.41  # C2
        bass_wave = np.sin(2 * np.pi * bass_freq * t) * 0.1
        
        # Drum pattern (kick op elke beat)
        kick_pattern = np.zeros_like(t)
        for i in range(int(duration_seconds * beat_freq / 60)):
            beat_time = i * beat_interval
            if beat_time < duration_seconds:
                start_idx = int(beat_time * sample_rate)
                end_idx = min(start_idx + int(0.1 * sample_rate), len(kick_pattern))
                if end_idx > start_idx:
                    kick_pattern[start_idx:end_idx] = np.sin(2 * np.pi * 40 * t[start_idx:end_idx]) * 0.15
        
        # Combineer
        instrumental_wave = bass_wave + kick_pattern
        
        # Normaliseer
        max_val = np.max(np.abs(instrumental_wave))
        if max_val > 0:
            instrumental_wave = instrumental_wave / max_val * 0.3  # 30% volume
        
        # Converteer naar AudioSegment
        instrumental_array = (instrumental_wave * 32767).astype(np.int16)
        instrumental_audio = AudioSegment(
            instrumental_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Maak stereo
        instrumental_audio = instrumental_audio.set_channels(2)
        
        return instrumental_audio
        
    except Exception as e:
        print(f"[!] Fout bij instrumentale generatie: {e}")
        return None


@app.route("/resultaat")
def resultaat():
    return send_from_directory(".", "live_vertaal.html")

@app.route("/song-generator")
def song_generator():
    """Route voor de AI liedjes generator interface."""
    return send_from_directory(".", "song_generator.html")


# -------------------- BEELDGENERATIE MET MODELLAB FLUX 2 PRO --------------------
def _generate_image_with_modelslab(prompt: str, width: int = 1024, height: int = 1024) -> Optional[str]:
    """
    Genereer een afbeelding met ModelsLab Flux 2 Pro Text To Image API.
    Retourneert het pad naar het opgeslagen afbeeldingbestand, of None bij fout.
    
    Let op: De API endpoint en request format kunnen verschillen per ModelsLab API versie.
    Controleer de ModelsLab documentatie voor de exacte endpoint en parameters.
    """
    if not modelslab_api_key:
        print("[!] ModelsLab API key niet beschikbaar voor beeldgeneratie")
        return None
    
    try:
        # ModelsLab API endpoint - mogelijk aanpassen volgens ModelsLab documentatie
        # Voorbeeld: https://api.modelslab.com/v1/images/generations
        # Of: https://modelslab.com/api/v1/text2img
        api_url = os.getenv("MODELLAB_API_URL", "https://api.modelslab.com/v1/images/generations")
        
        headers = {
            "Authorization": f"Bearer {modelslab_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "flux-2-pro",
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_images": 1
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Haal de afbeelding URL of base64 data op (afhankelijk van API response format)
        if "data" in result and len(result["data"]) > 0:
            image_data = result["data"][0]
            
            # Als de API een URL retourneert
            if "url" in image_data:
                image_url = image_data["url"]
                # Download de afbeelding
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                image_bytes = img_response.content
            # Als de API base64 data retourneert
            elif "b64_json" in image_data:
                image_bytes = base64.b64decode(image_data["b64_json"])
            else:
                print("[!] Onbekend response format van ModelsLab API")
                return None
            
            # Sla de afbeelding op in een tijdelijk bestand
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_bytes)
                return tmp.name
        
        print("[!] Geen afbeelding data in ModelsLab API response")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"[!] Fout bij ModelsLab API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"[!] API error details: {error_detail}")
            except:
                print(f"[!] API error response: {e.response.text}")
        return None
    except Exception as e:
        print(f"[!] Onverwachte fout bij beeldgeneratie: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    """
    Genereer een afbeelding met ModelsLab Flux 2 Pro Text To Image.
    Accepteert een text prompt en genereert een afbeelding.
    """
    if not modelslab_api_key:
        return jsonify({"error": "ModelsLab API key niet geconfigureerd"}), 503
    
    try:
        prompt = request.form.get("prompt") or request.json.get("prompt") if request.is_json else None
        if not prompt:
            return jsonify({"error": "Geen prompt opgegeven"}), 400
        
        width = int(request.form.get("width", 1024) or request.json.get("width", 1024) if request.is_json else 1024)
        height = int(request.form.get("height", 1024) or request.json.get("height", 1024) if request.is_json else 1024)
        
        # Valideer dimensies
        if width < 64 or width > 2048 or height < 64 or height > 2048:
            return jsonify({"error": "Afmetingen moeten tussen 64 en 2048 pixels zijn"}), 400
        
        print(f"[Image Generation] Genereer afbeelding voor prompt: {prompt[:50]}...")
        image_path = _generate_image_with_modelslab(prompt, width=width, height=height)
        
        if not image_path:
            return jsonify({"error": "Kon afbeelding niet genereren"}), 500
        
        @after_this_request
        def cleanup_file(response):
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"[!] Kon afbeeldingbestand niet verwijderen: {e}")
            return response
        
        return send_file(
            image_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
    except ValueError as e:
        return jsonify({"error": f"Ongeldige parameter: {str(e)}"}), 400
    except Exception as e:
        print(f"[!] Fout bij image generatie: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Fout bij image generatie: {str(e)}"}), 500

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)




































