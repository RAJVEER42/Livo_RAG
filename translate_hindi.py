"""
translate_hindi.py
------------------
Translates Hindi transcript JSONs to English and saves them as
  transcripts/<key>_translated.json

Only needed for Hindi videos. English videos are skipped.
Uses deep-translator (free, no API key).

Usage:
  python translate_hindi.py

Future use:
  - Add any Hindi/other-language video key to HINDI_VIDEOS below
  - Works for any language supported by GoogleTranslator source codes
    e.g. 'hi', 'ta', 'te', 'bn', 'mr'
"""

import json
import os
import time
from deep_translator import GoogleTranslator

TRANSCRIPTS_DIR = "transcripts"
BATCH_SIZE      = 30     # segments per translation request (keep under free-tier limits)
SLEEP_BETWEEN   = 0.3    # seconds between batches (be polite to free API)

HINDI_VIDEOS = [
    "v3_deep_learning_hi",
    "v4_ml_dl_hi",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def translate_segments(segments: list[dict], source_lang: str = "hi") -> list[dict]:
    """
    Translate all segment texts from source_lang → English.
    Batches segments to stay within free-tier request limits.
    Preserves all original fields; adds 'text_original' with source text.
    """
    translator   = GoogleTranslator(source=source_lang, target="en")
    translated   = []
    total        = len(segments)
    SEPARATOR    = " SPLITHERE "  # simple token unlikely to appear in ML content

    for batch_start in range(0, total, BATCH_SIZE):
        batch = segments[batch_start : batch_start + BATCH_SIZE]
        combined = SEPARATOR.join(seg["text"] for seg in batch)

        try:
            result = translator.translate(combined)
            parts  = result.split("SPLITHERE")

            for j, seg in enumerate(batch):
                seg_copy = seg.copy()
                seg_copy["text_original"] = seg["text"]
                seg_copy["text"] = parts[j].strip() if j < len(parts) else seg["text"]
                translated.append(seg_copy)

        except Exception as e:
            print(f"    Batch {batch_start // BATCH_SIZE + 1} failed ({e}) — keeping original text")
            for seg in batch:
                seg_copy = seg.copy()
                seg_copy["text_original"] = seg["text"]
                translated.append(seg_copy)

        # Progress
        done = min(batch_start + BATCH_SIZE, total)
        print(f"    {done}/{total} segments translated...", end="\r")
        time.sleep(SLEEP_BETWEEN)

    print()  # newline after progress
    return translated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for key in HINDI_VIDEOS:
        src_path = os.path.join(TRANSCRIPTS_DIR, f"{key}_raw.json")
        out_path = os.path.join(TRANSCRIPTS_DIR, f"{key}_translated.json")

        if os.path.exists(out_path):
            print(f"Skipping {key} — translated file already exists.")
            print(f"  Delete {out_path} to re-translate.\n")
            continue

        print(f"Translating: {key}")
        with open(src_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        print(f"  Segments: {len(segments)}")
        translated = translate_segments(segments)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {out_path}\n")

    print("Done. Re-run annotate.py to get Hindi video annotations.")


if __name__ == "__main__":
    main()
