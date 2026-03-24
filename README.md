# Livo_RAG

A gold-standard QA evaluation dataset for RAG systems, built from four ML/AI educational YouTube videos.

## Videos Used

| Key | Title | Language |
|---|---|---|
| v1 | 3Blue1Brown — But what is a Neural Network? | English |
| v2 | 3Blue1Brown — Transformers, the tech behind LLMs | English |
| v3 | CampusX — What is Deep Learning? | Hindi (translated) |
| v4 | CodeWithHarry — All About ML & Deep Learning | Hindi (translated) |

## Pipeline

```
pull_transcripts.py   →  transcripts/
translate_hindi.py    →  transcripts/*_translated.json
annotate.py           →  annotations/
generate_qa_pairs.py  →  qa_pairs/
evaluate_rag.py       →  eval_results/retrieval_report.txt
evaluate_answers.py   →  eval_results/answer_quality_report.txt
```

## Setup

```bash
pip install youtube-transcript-api deep-translator anthropic scikit-learn rank-bm25 sentence-transformers claude-agent-sdk anyio
```

## Usage

```bash
python pull_transcripts.py     # fetch transcripts
python translate_hindi.py      # translate Hindi videos
python annotate.py             # keyword annotation
python generate_qa_pairs.py    # generate QA pairs
python evaluate_rag.py         # retrieval evaluation
python evaluate_answers.py     # answer quality evaluation
```

## Results

**29 QA pairs** across 5 failure modes: `semantic_precision`, `multi_hop`, `negation_misconception`, `contrast`, `taxonomy`.

| Retriever | Recall@1 | MRR |
|---|---|---|
| TF-IDF | 0.724 | 0.790 |
| BM25 | 0.793 | 0.830 |
| Dense (MiniLM) | 0.724 | 0.807 |

See `SUBMISSION.md` for the 5 selected QA pairs and methodology.
