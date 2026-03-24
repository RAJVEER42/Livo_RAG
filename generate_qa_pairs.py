"""
generate_qa_pairs.py
--------------------
Step 3b + Step 4

Reads annotations/annotations.json and generates QA pairs for each
concept group using an LLM, selecting the best segment per group.

Output:
  qa_pairs/qa_pairs.json   — structured QA dataset
  qa_pairs/qa_pairs.txt    — human-readable version

Usage:
  python generate_qa_pairs.py
"""

import anyio
import json
import os
import re
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

ANNOTATIONS_PATH = "annotations/annotations.json"
OUTPUT_DIR       = "qa_pairs"

SYSTEM_PROMPT = """You are building a gold-standard QA evaluation dataset for RAG systems.
Given a transcript segment, generate one high-quality question-answer pair that tests
whether a RAG system can retrieve and correctly answer from this exact segment.

Rules:
- The question must be naturally phrased, as a real user would ask it.
- The question must be answerable ONLY from the provided segment text (not general knowledge).
- The ideal_answer must be a direct, factual answer grounded in the segment text.
- Keep the ideal_answer concise (1-3 sentences).
- The question should specifically probe the failure_mode described.
- Do NOT start questions with "According to the video" or "In this transcript".

Failure mode guidance:
  semantic_precision     → question requires precise technical distinction (e.g. what exactly bias does)
  negation_misconception → question has a common wrong belief baked in (e.g. "isn't sigmoid still used?")
  multi_hop              → question requires connecting two pieces of info from the segment
  contrast               → question asks to compare/contrast two things mentioned
  taxonomy               → question asks about category/relationship (e.g. "is X a subset of Y?")

Respond with ONLY a JSON object, no markdown fences:
{
  "question": "...",
  "ideal_answer": "...",
  "reasoning": "one sentence: why this question tests the failure mode"
}"""


def select_best_block(blocks: list[dict]) -> dict:
    """Pick the block with the most content (longest text)."""
    return max(blocks, key=lambda b: len(b["text"]))


async def generate_qa_pair(segment: dict, failure_mode: str, concept_label: str) -> dict:
    """Generate a QA pair for a single segment using an LLM."""
    prompt = f"""{SYSTEM_PROMPT}

Concept group: {concept_label}
Failure mode: {failure_mode}
Video timestamp: {segment['timestamp_start']} → {segment['timestamp_end']}
Matched keywords: {', '.join(segment['matched'])}

Segment text:
\"\"\"{segment['text'][:3000]}\"\"\"

Generate a QA pair that tests the {failure_mode} failure mode for this segment.
Respond with ONLY a JSON object."""

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=[],
        ),
    ):
        if isinstance(message, ResultMessage):
            result_text = message.result

    # Strip markdown code fences if model adds them
    text = result_text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    return json.loads(text)


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    video_meta = {
        "v1_neural_network":   "3B1B — But what is a Neural Network?",
        "v2_transformers":     "3B1B — Transformers, the tech behind LLMs",
        "v3_deep_learning_hi": "CampusX — What is Deep Learning? (Hindi, translated)",
        "v4_ml_dl_hi":         "CodeWithHarry — All About ML & Deep Learning (Hindi, translated)",
    }

    qa_pairs    = []
    total_pairs = 0

    for video_key, groups in annotations.items():
        video_title = video_meta.get(video_key, video_key)
        print(f"\n{'─' * 60}")
        print(f"Video: {video_title}")
        print(f"{'─' * 60}")

        for concept_label, data in groups.items():
            failure_mode = data["failure_mode"]
            blocks       = data["blocks"]

            if not blocks:
                print(f"  [{concept_label}] — no blocks, skipping")
                continue

            best_block = select_best_block(blocks)
            print(f"  [{concept_label}] failure_mode={failure_mode}  "
                  f"blocks={len(blocks)}  using block @ {best_block['timestamp_start']} ...")

            try:
                qa = await generate_qa_pair(best_block, failure_mode, concept_label)

                entry = {
                    "id":               f"{video_key}__{concept_label}",
                    "video_key":        video_key,
                    "video_title":      video_title,
                    "concept_label":    concept_label,
                    "failure_mode":     failure_mode,
                    "timestamp_start":  best_block["timestamp_start"],
                    "timestamp_end":    best_block["timestamp_end"],
                    "source_text":      best_block["text"],
                    "matched_keywords": best_block["matched"],
                    "question":         qa["question"],
                    "ideal_answer":     qa["ideal_answer"],
                    "reasoning":        qa.get("reasoning", ""),
                }

                qa_pairs.append(entry)
                total_pairs += 1

                print(f"    Q: {qa['question'][:90]}")
                print(f"    A: {qa['ideal_answer'][:90]}")

            except Exception as e:
                print(f"    ERROR: {e}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "qa_pairs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # Save readable TXT
    txt_path = os.path.join(OUTPUT_DIR, "qa_pairs.txt")
    lines = [
        "RAG EVALUATION QA PAIRS",
        f"Total pairs: {total_pairs}",
        "=" * 70,
        "",
    ]
    for i, p in enumerate(qa_pairs, 1):
        lines += [
            f"\n[{i:02d}]  id={p['id']}",
            f"  Video      : {p['video_title']}",
            f"  Timestamp  : {p['timestamp_start']} → {p['timestamp_end']}",
            f"  Concept    : {p['concept_label']}",
            f"  Failure    : {p['failure_mode']}",
            f"  Keywords   : {', '.join(p['matched_keywords'])}",
            f"",
            f"  QUESTION   : {p['question']}",
            f"  ANSWER     : {p['ideal_answer']}",
            f"  REASONING  : {p['reasoning']}",
            f"",
            f"  SOURCE     : {p['source_text'][:200]}...",
            "",
        ]
    lines += ["=" * 70, "END"]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n\nDone. Generated {total_pairs} QA pairs.")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    print(f"\nNext: run evaluate_rag.py for retrieval evaluation.")


if __name__ == "__main__":
    anyio.run(main)
