"""
evaluate_answers.py
-------------------
Step 6 — Answer Quality Evaluation (LLM-as-judge)

For each QA pair:
  1. Retrieve top-1 chunk from each retriever (TF-IDF, BM25, Dense)
  2. Generate an answer from each retrieved chunk
  3. Judge each generated answer against the gold ideal_answer

Scores per answer:
  correctness   1–5  (does it correctly answer the question per the gold?)
  faithfulness  1–3  (is it grounded in the retrieved chunk, not hallucinated?)
  coverage      1–3  (does it cover the key points of the gold answer?)

Output:
  eval_results/answer_quality_report.txt
  eval_results/answer_quality_results.json

Usage:
  python evaluate_answers.py
"""

import anyio
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

# ── Config ────────────────────────────────────────────────────────────────────

QA_PATH         = "qa_pairs/qa_pairs.json"
TRANSCRIPTS_DIR = "transcripts"
OUTPUT_DIR      = "eval_results"
CHUNK_SIZE      = 10
CHUNK_OVERLAP   = 3

VIDEO_FILES = {
    "v1_neural_network":   "v1_neural_network_raw.json",
    "v2_transformers":     "v2_transformers_raw.json",
    "v3_deep_learning_hi": "v3_deep_learning_hi_translated.json",
    "v4_ml_dl_hi":         "v4_ml_dl_hi_translated.json",
}

# ── Corpus (same as evaluate_rag.py) ─────────────────────────────────────────

def load_segments(video_key):
    path = os.path.join(TRANSCRIPTS_DIR, VIDEO_FILES[video_key])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_chunks(segments, video_key):
    chunks, i = [], 0
    while i < len(segments):
        window = segments[i : i + CHUNK_SIZE]
        text   = " ".join(s.get("text", "") for s in window)
        chunks.append({
            "chunk_id":  f"{video_key}__c{i}",
            "video_key": video_key,
            "timestamp": window[0]["timestamp"],
            "text":      text,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_corpus():
    video_keys = list(VIDEO_FILES.keys())
    all_chunks = []
    for vk in video_keys:
        segs = load_segments(vk)
        all_chunks.extend(make_chunks(segs, vk))
    return all_chunks

# ── Retrievers ────────────────────────────────────────────────────────────────

def build_retrievers(corpus):
    texts = [c["text"] for c in corpus]

    # TF-IDF
    vec    = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform(texts)

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)

    # Dense
    print("  Loading sentence-transformer...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding corpus...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    def retrieve_tfidf(q, k=1):
        q_vec  = vec.transform([q])
        scores = cosine_similarity(q_vec, matrix).flatten()
        return [corpus[i] for i in np.argsort(scores)[::-1][:k]]

    def retrieve_bm25(q, k=1):
        scores = bm25.get_scores(q.lower().split())
        return [corpus[i] for i in np.argsort(scores)[::-1][:k]]

    def retrieve_dense(q, k=1):
        q_emb  = model.encode([q])
        scores = cosine_similarity(q_emb, embeddings).flatten()
        return [corpus[i] for i in np.argsort(scores)[::-1][:k]]

    return {
        "TF-IDF": retrieve_tfidf,
        "BM25":   retrieve_bm25,
        "Dense":  retrieve_dense,
    }

# ── LLM judge prompt ─────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system.

Given:
  - A question
  - The gold ideal answer
  - Retrieved contexts and generated answers from 3 different retrievers

For each retriever:
  1. Generate a concise answer (2-3 sentences) ONLY using information from that retriever's context.
     If the context doesn't contain enough information, say "The retrieved context does not contain enough information to answer this."
  2. Score your generated answer on three dimensions:
       correctness  : 1-5  (1=completely wrong, 3=partially correct, 5=fully correct vs gold)
       faithfulness : 1-3  (1=hallucinated, 2=mostly grounded, 3=fully grounded in context)
       coverage     : 1-3  (1=misses key points, 2=covers some, 3=covers all key points of gold)

Respond with ONLY a JSON object in this exact format (no markdown):
{
  "TF-IDF": {
    "generated_answer": "...",
    "correctness": <1-5>,
    "faithfulness": <1-3>,
    "coverage": <1-3>,
    "reasoning": "one sentence"
  },
  "BM25": {
    "generated_answer": "...",
    "correctness": <1-5>,
    "faithfulness": <1-3>,
    "coverage": <1-3>,
    "reasoning": "one sentence"
  },
  "Dense": {
    "generated_answer": "...",
    "correctness": <1-5>,
    "faithfulness": <1-3>,
    "coverage": <1-3>,
    "reasoning": "one sentence"
  }
}"""


async def judge_one(qa, contexts):
    """Generate and judge answers for all 3 retrievers in a single LLM call."""
    context_block = "\n\n".join(
        f"--- {name} CONTEXT (chunk {ctx['chunk_id']}, timestamp {ctx['timestamp']}) ---\n{ctx['text'][:1500]}"
        for name, ctx in contexts.items()
    )

    prompt = f"""{JUDGE_PROMPT}

QUESTION: {qa['question']}

GOLD IDEAL ANSWER: {qa['ideal_answer']}

{context_block}

Now generate and score answers for all 3 retrievers. Output ONLY JSON."""

    result = ""
    async for msg in query(prompt=prompt, options=ClaudeAgentOptions(allowed_tools=[])):
        if isinstance(msg, ResultMessage):
            result = msg.result

    text = result.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    return json.loads(text)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(scores_list):
    """Aggregate a list of {correctness, faithfulness, coverage} dicts."""
    if not scores_list:
        return {}
    return {
        "correctness_avg":  round(sum(s["correctness"]  for s in scores_list) / len(scores_list), 2),
        "faithfulness_avg": round(sum(s["faithfulness"] for s in scores_list) / len(scores_list), 2),
        "coverage_avg":     round(sum(s["coverage"]     for s in scores_list) / len(scores_list), 2),
        "correct_5_pct":    round(sum(1 for s in scores_list if s["correctness"] == 5) / len(scores_list), 2),
        "n":                len(scores_list),
    }

# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results, output_path):
    retriever_names = ["TF-IDF", "BM25", "Dense"]

    lines = [
        "ANSWER QUALITY EVALUATION REPORT (LLM-as-Judge)",
        "=" * 70,
        "Scores: correctness 1-5 | faithfulness 1-3 | coverage 1-3",
        f"QA pairs evaluated: {len(all_results)}",
        "",
    ]

    # ── Overall metrics per retriever
    lines += ["\nOVERALL METRICS", "─" * 50]
    header = f"  {'Retriever':<10}  {'Correct(avg)':>12}  {'Faithful(avg)':>13}  {'Coverage(avg)':>13}  {'Score=5(%)':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for rname in retriever_names:
        scores = [r["scores"][rname] for r in all_results if rname in r["scores"]]
        m = compute_metrics(scores)
        if m:
            lines.append(
                f"  {rname:<10}  {m['correctness_avg']:>12.2f}  "
                f"{m['faithfulness_avg']:>13.2f}  {m['coverage_avg']:>13.2f}  "
                f"{m['correct_5_pct']:>10.0%}"
            )

    # ── Per failure mode
    lines += ["", "\nPER FAILURE MODE — CORRECTNESS AVG", "─" * 50]
    by_fm = defaultdict(list)
    for r in all_results:
        by_fm[r["failure_mode"]].append(r)

    header2 = f"  {'Failure Mode':<25}  {'TF-IDF':>7}  {'BM25':>7}  {'Dense':>7}  {'n':>4}"
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))

    for fm, items in sorted(by_fm.items()):
        row = f"  {fm:<25}"
        for rname in retriever_names:
            scores = [i["scores"][rname]["correctness"] for i in items if rname in i["scores"]]
            avg = sum(scores) / len(scores) if scores else 0
            row += f"  {avg:>7.2f}"
        row += f"  {len(items):>4}"
        lines.append(row)

    # ── Per question breakdown
    lines += ["", "\nPER-QUESTION BREAKDOWN", "─" * 70]
    for r in all_results:
        lines.append(f"\n  [{r['failure_mode']:<25}]  {r['question'][:65]}")
        for rname in retriever_names:
            if rname not in r["scores"]:
                continue
            s = r["scores"][rname]
            bar = "█" * s["correctness"] + "░" * (5 - s["correctness"])
            lines.append(
                f"    {rname:<8}  C={s['correctness']}/5 [{bar}]  "
                f"F={s['faithfulness']}/3  Cov={s['coverage']}/3  "
                f"| {s['reasoning'][:60]}"
            )
            lines.append(f"             → {s['generated_answer'][:100]}")

    lines += ["", "=" * 70, "END"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building corpus...")
    corpus = build_corpus()
    print(f"  Total chunks: {len(corpus)}")

    print("\nBuilding retrievers...")
    retrievers = build_retrievers(corpus)

    with open(QA_PATH) as f:
        qa_pairs = json.load(f)

    print(f"\nEvaluating {len(qa_pairs)} QA pairs...\n")

    all_results = []

    for i, qa in enumerate(qa_pairs, 1):
        print(f"[{i:02d}/{len(qa_pairs)}] {qa['id']}")

        # Retrieve top-1 chunk from each retriever
        contexts = {
            name: retrieve(qa["question"], 1)[0]
            for name, retrieve in retrievers.items()
        }

        try:
            judgements = await judge_one(qa, contexts)

            result = {
                "id":           qa["id"],
                "question":     qa["question"],
                "ideal_answer": qa["ideal_answer"],
                "failure_mode": qa["failure_mode"],
                "concept":      qa["concept_label"],
                "video_key":    qa["video_key"],
                "scores":       judgements,
                "contexts":     {k: v["chunk_id"] for k, v in contexts.items()},
            }
            all_results.append(result)

            # Quick print
            for rname in ["TF-IDF", "BM25", "Dense"]:
                if rname in judgements:
                    s = judgements[rname]
                    print(f"  {rname:<8} C={s['correctness']}/5  F={s['faithfulness']}/3  Cov={s['coverage']}/3")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    # Save
    report_path = os.path.join(OUTPUT_DIR, "answer_quality_report.txt")
    write_report(all_results, report_path)

    json_path = os.path.join(OUTPUT_DIR, "answer_quality_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {report_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    anyio.run(main)
