"""
pull_transcripts.py
-------------------
Pulls transcripts for all 4 target videos and saves them as:
  - <name>_raw.json     : full timestamped segments (used for precise sourcing)
  - <name>_full.txt     : plain readable text (used for annotation in Step 3)

Usage:
  python pull_transcripts.py

No API key needed. Uses YouTube's built-in caption system.

Future use:
  - Add any video_id + name to VIDEOS dict to pull transcripts for new projects
  - set TRANSLATE_HINDI=True to auto-translate Hindi transcripts to English
    (uses deep-translator, free, no key needed)
"""

import json
import os
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from deep_translator import GoogleTranslator

# ── Config ────────────────────────────────────────────────────────────────────

VIDEOS = {
    "v1_neural_network":   {"id": "aircAruvnKk", "lang": "en", "title": "3B1B — But what is a Neural Network?"},
    "v2_transformers":     {"id": "wjZofJX0v4M", "lang": "en", "title": "3B1B — Transformers, the tech behind LLMs"},
    "v3_deep_learning_hi": {"id": "fHF22Wxuyw4", "lang": "hi", "title": "CampusX — What is Deep Learning? (Hindi)"},
    "v4_ml_dl_hi":         {"id": "C6YtPJxNULA", "lang": "hi", "title": "CodeWithHarry — All About ML & Deep Learning (Hindi)"},
}

TRANSLATE_HINDI = False   # flip to True if you want Hindi auto-translated to English
OUTPUT_DIR = "transcripts"

# ── Helpers ───────────────────────────────────────────────────────────────────

def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds → human-readable MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fetch_transcript(video_id: str, preferred_lang: str) -> list[dict]:
    """
    Fetch transcript segments. Tries preferred language first,
    then falls back to any available language.
    Returns list of {text, start, duration, timestamp} dicts.

    Note: youtube-transcript-api >= 1.x requires instantiation — api = YouTubeTranscriptApi()
    """
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=[preferred_lang])
    except NoTranscriptFound:
        # fallback: grab whatever language is available
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript([preferred_lang])
        fetched = transcript.fetch()

    # Enrich each segment with a human-readable timestamp
    # Note: v1.x returns FetchedTranscriptSnippet dataclass objects (not dicts)
    # Access via .text / .start / .duration attributes
    enriched = []
    for seg in fetched:
        enriched.append({
            "text":      seg.text,
            "start":     seg.start,
            "duration":  seg.duration,
            "timestamp": seconds_to_timestamp(seg.start),
        })
    return enriched


def translate_segments(segments: list[dict]) -> list[dict]:
    """
    Translate segment text from Hindi to English using deep-translator.
    Batches requests to avoid hitting free-tier limits.
    Only called when TRANSLATE_HINDI=True.
    """
    translator = GoogleTranslator(source="hi", target="en")
    BATCH_SIZE = 40  # translate 40 segments at a time

    translated = []
    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i : i + BATCH_SIZE]
        combined = "\n||||\n".join(s["text"] for s in batch)
        try:
            result = translator.translate(combined)
            parts = result.split("\n||||\n")
            for j, seg in enumerate(batch):
                seg_copy = seg.copy()
                seg_copy["text_original_hi"] = seg["text"]
                seg_copy["text"] = parts[j].strip() if j < len(parts) else seg["text"]
                translated.append(seg_copy)
        except Exception as e:
            print(f"  Translation batch {i//BATCH_SIZE + 1} failed: {e} — keeping original")
            translated.extend(batch)

    return translated


def save_raw_json(segments: list[dict], path: str):
    """Save full timestamped segments as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def save_readable_txt(segments: list[dict], title: str, path: str):
    """
    Save a human-readable transcript with timestamps every N seconds.
    Format:
      [0:00] text text text
      [0:30] text text text
    Makes manual annotation in Step 3 much easier.
    """
    STAMP_EVERY = 30  # insert timestamp marker every 30 seconds

    lines = [f"TRANSCRIPT: {title}", "=" * 60, ""]
    last_stamped = -STAMP_EVERY

    for seg in segments:
        if seg["start"] - last_stamped >= STAMP_EVERY:
            lines.append(f"\n[{seg['timestamp']}]")
            last_stamped = seg["start"]
        lines.append(seg["text"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving to ./{OUTPUT_DIR}/\n")

    for name, meta in VIDEOS.items():
        print(f"Fetching: {meta['title']}")
        print(f"  Video ID : {meta['id']}")

        try:
            segments = fetch_transcript(meta["id"], meta["lang"])
            print(f"  Segments : {len(segments)}")

            # Optionally translate Hindi
            if TRANSLATE_HINDI and meta["lang"] == "hi":
                print(f"  Translating Hindi → English ...")
                segments = translate_segments(segments)

            # Save both formats
            raw_path = os.path.join(OUTPUT_DIR, f"{name}_raw.json")
            txt_path = os.path.join(OUTPUT_DIR, f"{name}_full.txt")

            save_raw_json(segments, raw_path)
            save_readable_txt(segments, meta["title"], txt_path)

            duration_min = segments[-1]["start"] / 60 if segments else 0
            print(f"  Duration : ~{duration_min:.1f} min")
            print(f"  Saved    : {raw_path}")
            print(f"  Saved    : {txt_path}")

        except TranscriptsDisabled:
            print(f"  ERROR: Transcripts are disabled for this video.")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print("Done. All transcripts saved.")
    print(f"\nNext: open transcripts/*.txt and annotate key segments (Step 3).")


if __name__ == "__main__":
    main()
