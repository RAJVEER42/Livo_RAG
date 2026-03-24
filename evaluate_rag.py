"""
evaluate_rag.py
---------------
Step 5 — RAG Retrieval Evaluation

Builds a corpus from all 4 transcript JSONs, indexes it with three retrievers
(TF-IDF, BM25, Dense/sentence-transformers), then evaluates retrieval quality
against the gold QA pairs.

Metrics computed per retriever and per failure mode:
  Recall@k  (k = 1, 3, 5) — did the gold chunk appear in the top-k results?
  MRR       — mean reciprocal rank of the first gold hit
  Precision@1 — is the top result correct?

Output:
  eval_results/retrieval_report.txt   — human-readable report
  eval_results/retrieval_results.json — raw per-question scores

Usage:
  python evaluate_rag.py
"""

import json
import os
import re
import math
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

QA_PATH       = "qa_pairs/qa_pairs.json"
TRANSCRIPTS_DIR = "transcripts"
OUTPUT_DIR    = "eval_results"

CHUNK_SIZE    = 10   # segments per chunk
CHUNK_OVERLAP = 3    # overlap between consecutive chunks
TOP_K         = [1, 3, 5]

# Gold match: what fraction of gold source text must overlap with retrieved chunk
OVERLAP_THRESHOLD = 0.30   # 30% of gold text words must appear in chunk

VIDEO_FILES = {
    "v1_neural_network":   "v1_neural_network_raw.json",
    "v2_transformers":     "v2_transformers_raw.json",
    "v3_deep_learning_hi": "v3_deep_learning_hi_translated.json",
    "v4_ml_dl_hi":         "v4_ml_dl_hi_translated.json",
}

# ── Corpus builder ────────────────────────────────────────────────────────────

def load_segments(video_key: str) -> list[dict]:
    fname = VIDEO_FILES[video_key]
    path  = os.path.join(TRANSCRIPTS_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_chunks(segments: list[dict], video_key: str, video_title: str) -> list[dict]:
    """Slide a window over segments to produce overlapping text chunks."""
    chunks = []
    i = 0
    while i < len(segments):
        window = segments[i : i + CHUNK_SIZE]
        text   = " ".join(s.get("text", "") for s in window)
        chunks.append({
            "chunk_id":    f"{video_key}__c{i}",
            "video_key":   video_key,
            "video_title": video_title,
            "start_sec":   window[0]["start"],
            "end_sec":     window[-1]["start"] + window[-1].get("duration", 0),
            "timestamp":   window[0]["timestamp"],
            "text":        text,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_corpus() -> list[dict]:
    video_titles = {
        "v1_neural_network":   "3B1B — But what is a Neural Network?",
        "v2_transformers":     "3B1B — Transformers, the tech behind LLMs",
        "v3_deep_learning_hi": "CampusX — What is Deep Learning? (Hindi, translated)",
        "v4_ml_dl_hi":         "CodeWithHarry — All About ML & Deep Learning (Hindi, translated)",
    }
    all_chunks = []
    for vk, title in video_titles.items():
        segs = load_segments(vk)
        chunks = make_chunks(segs, vk, title)
        all_chunks.extend(chunks)
        print(f"  {vk}: {len(segs)} segments → {len(chunks)} chunks")
    return all_chunks

# ── Gold match ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def is_gold_match(chunk_text: str, gold_text: str) -> bool:
    """True if chunk contains ≥ OVERLAP_THRESHOLD fraction of gold text words."""
    gold_words  = tokenize(gold_text)
    chunk_words = tokenize(chunk_text)
    if not gold_words:
        return False
    overlap = len(gold_words & chunk_words) / len(gold_words)
    return overlap >= OVERLAP_THRESHOLD

# ── Retrievers ────────────────────────────────────────────────────────────────

class TFIDFRetriever:
    name = "TF-IDF"

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        texts = [c["text"] for c in corpus]
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vec.fit_transform(texts)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q_vec  = self.vec.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in top_idx]


class BM25Retriever:
    name = "BM25"

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        tokenized = [c["text"].lower().split() for c in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in top_idx]


class DenseRetriever:
    name = "Dense (all-MiniLM-L6-v2)"

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        print("  Loading sentence-transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Encoding {len(corpus)} chunks...")
        texts = [c["text"] for c in corpus]
        self.embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q_emb   = self.model.encode([query])
        scores  = cosine_similarity(q_emb, self.embeddings).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in top_idx]

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(retriever, qa_pairs: list[dict], max_k: int) -> list[dict]:
    results = []
    for qa in qa_pairs:
        retrieved = retriever.retrieve(qa["question"], top_k=max_k)
        gold_text = qa["source_text"]

        hits = [is_gold_match(c["text"], gold_text) for c in retrieved]

        # Reciprocal rank
        rr = 0.0
        for rank, hit in enumerate(hits, 1):
            if hit:
                rr = 1.0 / rank
                break

        results.append({
            "id":           qa["id"],
            "failure_mode": qa["failure_mode"],
            "concept":      qa["concept_label"],
            "question":     qa["question"],
            "hits":         hits,
            "rr":           rr,
            "retrieved":    [c["chunk_id"] for c in retrieved],
        })
    return results


def compute_metrics(results: list[dict], k_list: list[int]) -> dict:
    metrics = {}
    for k in k_list:
        recall_at_k = [1.0 if any(r["hits"][:k]) else 0.0 for r in results]
        metrics[f"Recall@{k}"] = sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0
    metrics["MRR"]        = sum(r["rr"] for r in results) / len(results) if results else 0
    metrics["P@1"]        = sum(1 for r in results if r["hits"] and r["hits"][0]) / len(results) if results else 0
    return metrics


def compute_per_failure_mode(results: list[dict], k_list: list[int]) -> dict[str, dict]:
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["failure_mode"]].append(r)
    return {mode: compute_metrics(items, k_list) for mode, items in by_mode.items()}

# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(all_eval: dict, output_path: str, qa_pairs: list[dict], k_list: list[int]):
    lines = [
        "RAG RETRIEVAL EVALUATION REPORT",
        "=" * 70,
        f"Corpus: 4 videos, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}",
        f"Gold match threshold: {OVERLAP_THRESHOLD*100:.0f}% word overlap",
        f"QA pairs evaluated: {len(qa_pairs)}",
        "",
    ]

    for retriever_name, data in all_eval.items():
        lines += [
            f"\n{'─' * 70}",
            f"RETRIEVER: {retriever_name}",
            f"{'─' * 70}",
        ]

        # Overall metrics
        overall = data["overall"]
        lines.append("\nOverall metrics:")
        for metric, val in overall.items():
            lines.append(f"  {metric:<12} {val:.3f}")

        # Per failure mode
        lines.append("\nPer failure mode:")
        header = f"  {'Failure Mode':<25}" + "".join(f"  R@{k}" for k in k_list) + "   MRR   P@1"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for mode, m in sorted(data["per_failure_mode"].items()):
            row = f"  {mode:<25}"
            for k in k_list:
                row += f"  {m[f'Recall@{k}']:.2f}"
            row += f"  {m['MRR']:.2f}  {m['P@1']:.2f}"
            lines.append(row)

        # Per-question breakdown
        lines.append("\nPer-question results:")
        for r in data["raw"]:
            hit_str = "HIT " if r["rr"] > 0 else "MISS"
            first_rank = next((i+1 for i, h in enumerate(r["hits"]) if h), None)
            rank_str   = f"rank={first_rank}" if first_rank else "not found"
            lines.append(f"  [{hit_str}] [{r['failure_mode']:<25}] {rank_str:>12}  {r['question'][:65]}")

    lines += ["", "=" * 70, "END"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building corpus...")
    corpus = build_corpus()
    print(f"  Total chunks: {len(corpus)}\n")

    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"QA pairs loaded: {len(qa_pairs)}\n")

    retrievers = [
        TFIDFRetriever(corpus),
        BM25Retriever(corpus),
        DenseRetriever(corpus),
    ]

    all_eval = {}
    max_k = max(TOP_K)

    for retriever in retrievers:
        print(f"Evaluating {retriever.name}...")
        raw     = evaluate(retriever, qa_pairs, max_k)
        overall = compute_metrics(raw, TOP_K)
        per_fm  = compute_per_failure_mode(raw, TOP_K)

        all_eval[retriever.name] = {
            "overall":          overall,
            "per_failure_mode": per_fm,
            "raw":              raw,
        }

        print(f"  Recall@1={overall['Recall@1']:.3f}  "
              f"Recall@3={overall['Recall@3']:.3f}  "
              f"Recall@5={overall['Recall@5']:.3f}  "
              f"MRR={overall['MRR']:.3f}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "retrieval_report.txt")
    write_report(all_eval, report_path, qa_pairs, TOP_K)

    # Save raw JSON (exclude verbose raw hits for brevity)
    json_out = {
        rname: {
            "overall":          d["overall"],
            "per_failure_mode": d["per_failure_mode"],
            "questions": [
                {k: v for k, v in r.items() if k != "retrieved"}
                for r in d["raw"]
            ],
        }
        for rname, d in all_eval.items()
    }
    json_path = os.path.join(OUTPUT_DIR, "retrieval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {report_path}")
    print(f"Saved: {json_path}")
    print(f"\nNext: Step 6 — answer quality evaluation (LLM-as-judge).")


if __name__ == "__main__":
    main()
