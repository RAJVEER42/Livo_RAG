"""
annotate.py
-----------
Scans all 4 transcript JSONs for concept keywords and extracts
the surrounding context window with timestamps.

Output:
  - annotations/annotations.txt   : human-readable shortlist for Step 3
  - annotations/annotations.json  : structured data (for future programmatic use)

Usage:
  python annotate.py

Future use:
  - Drop any transcript JSON into transcripts/ and add its entry to VIDEOS
  - Edit CONCEPT_GROUPS to target different topic areas for new RAG eval projects
  - Increase CONTEXT_WINDOW to get more surrounding text per hit
  - Set MIN_GAP_SECONDS to control how close two hits need to be to merge
"""

import json
import os

# ── Config ────────────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = "transcripts"
OUTPUT_DIR      = "annotations"
CONTEXT_WINDOW  = 3     # segments to grab before and after each keyword hit
MIN_GAP_SECONDS = 20    # hits within this window get merged into one block

# Videos to scan — add new entries here for future projects
VIDEOS = [
    {
        "key":   "v1_neural_network",
        "title": "3B1B — But what is a Neural Network?",
        "lang":  "en",
    },
    {
        "key":   "v2_transformers",
        "title": "3B1B — Transformers, the tech behind LLMs",
        "lang":  "en",
    },
    {
        "key":   "v3_deep_learning_hi",
        "title": "CampusX — What is Deep Learning? (Hindi)",
        "lang":  "hi",
    },
    {
        "key":   "v4_ml_dl_hi",
        "title": "CodeWithHarry — All About ML & Deep Learning (Hindi)",
        "lang":  "hi",
    },
]

# ── Concept groups ────────────────────────────────────────────────────────────
# Each group has a label (the retrieval failure mode it targets) and keywords.
# Keywords are matched case-insensitively anywhere in the segment text.
# Hindi/English ML videos use English technical terms even mid-Hindi — so
# these English keywords work across all 4 videos.
#
# To reuse for a different domain: swap out the keyword lists below.

CONCEPT_GROUPS = [
    {
        "label":    "neuron_mechanics",
        "failure_mode": "semantic_precision",
        "keywords": [
            "weighted sum", "weight", "bias", "activation function",
            "sigmoid", "relu", "fires", "squish", "neuron activate",
            "dot product", "linear combination",
        ],
    },
    {
        "label":    "backprop_gradient",
        "failure_mode": "negation_misconception",
        "keywords": [
            "backprop", "back propagation", "gradient descent",
            "gradient", "chain rule", "derivative", "learning rate",
            "cost function", "loss function", "minimize",
        ],
    },
    {
        "label":    "depth_vs_shallow",
        "failure_mode": "negation_misconception",
        "keywords": [
            "universal approximat", "single layer", "one hidden layer",
            "shallow network", "why deep", "depth", "hierarchy of features",
            "hierarchical", "layer by layer", "abstract representation",
        ],
    },
    {
        "label":    "attention_mechanism",
        "failure_mode": "multi_hop",
        "keywords": [
            "query", "key", "value", "attention score", "self-attention",
            "self attention", "dot product attention", "softmax attention",
            "attend", "scaled dot", "multi-head", "multi head",
        ],
    },
    {
        "label":    "feature_engineering",
        "failure_mode": "contrast",
        "keywords": [
            "feature engineer", "feature extract", "manual feature",
            "hand-craft", "handcraft", "raw data", "automatically learn",
            "learns features", "representation learning",
        ],
    },
    {
        "label":    "ml_dl_taxonomy",
        "failure_mode": "taxonomy",
        "keywords": [
            "subset of", "subfield", "branch of",
            "artificial intelligence", "machine learning is",
            "deep learning is", "ai ml dl", "umbrella",
            "narrow ai", "ml vs dl", "difference between ml",
        ],
    },
    {
        "label":    "positional_encoding",
        "failure_mode": "multi_hop",
        "keywords": [
            "positional encoding", "position embedding", "order of words",
            "word order", "sequential", "position of",
        ],
    },
    {
        "label":    "layers_and_architecture",
        "failure_mode": "semantic_precision",
        "keywords": [
            "hidden layer", "input layer", "output layer",
            "number of layers", "784", "28 by 28", "16 neuron",
        ],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_transcript(key: str) -> list[dict]:
    """
    Load transcript for a video. For Hindi videos, uses the translated
    version (_translated.json) if it exists — falls back to raw otherwise.
    """
    translated_path = os.path.join(TRANSCRIPTS_DIR, f"{key}_translated.json")
    raw_path        = os.path.join(TRANSCRIPTS_DIR, f"{key}_raw.json")

    if os.path.exists(translated_path):
        path = translated_path
        print(f"    (using translated transcript)")
    else:
        path = raw_path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_hits(segments: list[dict], keywords: list[str]) -> list[int]:
    """Return indices of segments that contain any of the keywords."""
    hits = []
    for i, seg in enumerate(segments):
        text_lower = seg["text"].lower()
        if any(kw.lower() in text_lower for kw in keywords):
            hits.append(i)
    return hits


def expand_hits(hits: list[int], total: int, window: int) -> list[tuple[int, int]]:
    """
    Expand each hit index into a (start, end) range using the context window.
    Merge ranges that overlap or are within MIN_GAP_SECONDS of each other.
    Returns list of (start_idx, end_idx) tuples.
    """
    if not hits:
        return []

    ranges = []
    for h in hits:
        s = max(0, h - window)
        e = min(total - 1, h + window)
        ranges.append((s, e))

    # Merge overlapping/adjacent ranges
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


def build_block(segments: list[dict], start: int, end: int, matched_keywords: list[str]) -> dict:
    """Build a single annotation block from a range of segments."""
    block_segs = segments[start:end + 1]
    text = " ".join(s["text"] for s in block_segs)
    return {
        "timestamp_start": block_segs[0]["timestamp"],
        "timestamp_end":   block_segs[-1]["timestamp"],
        "start_seconds":   block_segs[0]["start"],
        "text":            text,
        "matched":         matched_keywords,
    }


def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── Core annotation logic ─────────────────────────────────────────────────────

def annotate_video(video: dict) -> dict:
    """
    Run all concept groups against a single video's transcript.
    Returns structured annotation results.
    """
    segments = load_transcript(video["key"])
    total    = len(segments)
    results  = {}

    for group in CONCEPT_GROUPS:
        hits = find_hits(segments, group["keywords"])
        if not hits:
            continue

        # Check which keywords actually matched (for display)
        matched = []
        for kw in group["keywords"]:
            for seg in segments:
                if kw.lower() in seg["text"].lower():
                    matched.append(kw)
                    break

        ranges = expand_hits(hits, total, CONTEXT_WINDOW)
        blocks = [build_block(segments, s, e, matched) for s, e in ranges]

        results[group["label"]] = {
            "failure_mode": group["failure_mode"],
            "hit_count":    len(hits),
            "blocks":       blocks,
        }

    return results


# ── Output formatters ─────────────────────────────────────────────────────────

def write_readable(all_results: dict, output_path: str):
    """Write the human-readable annotation file for Step 3."""
    lines = [
        "TRANSCRIPT ANNOTATIONS",
        "Keyword-extracted segments with timestamps for QA pair construction",
        "=" * 70,
        "",
    ]

    for video in VIDEOS:
        vkey    = video["key"]
        vtitle  = video["title"]
        vresult = all_results.get(vkey, {})

        lines.append(f"\n{'─' * 70}")
        lines.append(f"VIDEO: {vtitle}")
        lines.append(f"{'─' * 70}")

        if not vresult:
            lines.append("  (no keyword hits found)")
            continue

        for group_label, data in vresult.items():
            lines.append(f"\n  [{group_label}]  failure_mode={data['failure_mode']}  hits={data['hit_count']}")
            lines.append(f"  matched keywords: {', '.join(set(data['blocks'][0]['matched'])) if data['blocks'] else '—'}")
            lines.append("")

            for i, block in enumerate(data["blocks"], 1):
                ts = f"{block['timestamp_start']} → {block['timestamp_end']}"
                lines.append(f"    Block {i}  [{ts}]")
                # Wrap text at 80 chars for readability
                words = block["text"].split()
                line_buf = "      "
                for word in words:
                    if len(line_buf) + len(word) + 1 > 82:
                        lines.append(line_buf)
                        line_buf = "      " + word
                    else:
                        line_buf += (" " if line_buf.strip() else "") + word
                if line_buf.strip():
                    lines.append(line_buf)
                lines.append("")

        # Per-video summary
        lines.append(f"\n  SUMMARY for {vkey}:")
        for group_label, data in vresult.items():
            lines.append(f"    {group_label:30s} → {data['hit_count']} hits, {len(data['blocks'])} blocks")

    lines.append("\n" + "=" * 70)
    lines.append("END — Use this file to pick segments for Step 3 annotation")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_json(all_results: dict, output_path: str):
    """Write structured JSON — useful if you want to process annotations programmatically."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running keyword annotation across all 4 transcripts...\n")

    all_results = {}
    for video in VIDEOS:
        print(f"  Scanning: {video['title']}")
        result = annotate_video(video)
        all_results[video["key"]] = result

        hit_summary = {g: d["hit_count"] for g, d in result.items()}
        print(f"  Groups hit: {list(hit_summary.keys())}")
        print(f"  Hit counts: {hit_summary}\n")

    txt_path  = os.path.join(OUTPUT_DIR, "annotations.txt")
    json_path = os.path.join(OUTPUT_DIR, "annotations.json")

    write_readable(all_results, txt_path)
    write_json(all_results, json_path)

    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    print(f"\nNext: open annotations/annotations.txt and pick your 8-10 candidate segments (Step 3).")


if __name__ == "__main__":
    main()
