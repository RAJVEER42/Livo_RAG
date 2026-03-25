# Option B — Golden Dataset for RAG
### Submission by Rajveer Bishnoi

---

## QA Pairs

---

### Pair 1

**Question:**
What condition in the pixel values causes the weighted sum to be at its largest when trying to detect an edge, and what does that tell us about the role of the negative weights?

**Answer:**
The weighted sum is largest when the middle pixels are bright but the surrounding pixels are darker. The negative weights are assigned to the surrounding pixels precisely to create this contrast — so the sum peaks when there is a meaningful brightness difference between the centre region and its surroundings, which is the signature of an edge.

**Source:**
3Blue1Brown — *But what is a Neural Network?*
Timestamp: **9:27 → 10:32**
`youtube.com/watch?v=aircAruvnKk&t=567`

---

### Pair 2

**Question:**
Does using temperature zero in GPT-3 generation produce output based on backpropagated derivatives of the training gradient?

**Answer:**
No. In this context, "derivative" refers to the story being an unoriginal, clichéd imitation — specifically described as "a trite derivative of Goldilocks." Temperature zero simply means the model always selects the most statistically predictable next word. It has no connection to backpropagation or gradients.

**Source:**
3Blue1Brown — *Transformers, the tech behind LLMs*
Timestamp: **24:33 → 24:58**
`youtube.com/watch?v=wjZofJX0v4M&t=1473`

---

### Pair 3

**Question:**
What two things does a word embedding vector in a transformer encode beyond just the identity of the word itself?

**Answer:**
In a transformer, a word embedding vector encodes both the **position** of that word in the sequence and **contextual information** from surrounding words — meaning the vector has the capacity to "soak in context" as it passes through the model's layers.

**Source:**
3Blue1Brown — *Transformers, the tech behind LLMs*
Timestamp: **18:20 → 18:43**
`youtube.com/watch?v=wjZofJX0v4M&t=1100`

---

### Pair 4

**Question:**
What distinguishes automatic feature extraction from manual feature engineering in terms of where the process occurs in a pipeline?

**Answer:**
Automatic feature extraction takes place inside the model pipeline itself, whereas manual feature engineering is done externally before the model processes the data. This internal automatic process removes the need for hand-crafted features when the data conditions are appropriate.

**Source:**
CampusX — *What is Deep Learning?* (Hindi, auto-translated)
Timestamp: **16:48 → 17:04**
`youtube.com/watch?v=fHF22Wxuyw4&t=1008`

---

### Pair 5

**Question:**
If today's precipitation, humidity, and temperature values are fed into the system, what does the system produce as an output and what underlying process makes that possible?

**Answer:**
The system produces a prediction about whether it will rain today (a true/false outcome). This is made possible by machine learning, which learns from historical labelled examples of those input values paired with whether it actually rained — allowing the model to generalise to new inputs.

**Source:**
CodeWithHarry — *All About ML & Deep Learning* (Hindi, auto-translated)
Timestamp: **1:46 → 2:08**
`youtube.com/watch?v=C6YtPJxNULA&t=106`

---

## Methodology Note

### How questions were selected

- Built a keyword-annotation pipeline (`annotate.py`) that scanned all 4 transcripts for 8 concept groups (neuron mechanics, backprop, attention, positional encoding, feature engineering, taxonomy, layers, depth-vs-shallow), each tagged with a target retrieval failure mode.
- Generated 29 candidate QA pairs using Claude (via Claude Agent SDK — no API key needed) and evaluated each against three retrievers: TF-IDF, BM25, and a dense sentence-transformer model.
- The 5 pairs submitted were chosen to satisfy three criteria simultaneously:
  1. High answer-quality score (≥ 4/5 correctness, judged by LLM-as-judge) across all retrievers
  2. Coverage of at least 3 different videos and 3 different failure modes
  3. Self-contained answers — answerable purely from the transcript segment, no external knowledge needed

### How they were pulled from the material

1. Fetched raw transcripts using `youtube-transcript-api` (no API key needed).
2. Hindi transcripts (videos 3 & 4) were translated to English using `deep-translator` (Google Translate, free tier), producing `_translated.json` files with original Hindi preserved alongside English.
3. Scanned all segments for concept keywords; extracted context windows (± 3 segments around each hit) to produce candidate blocks with timestamps.
4. Claude generated a question + ideal answer for each block given the failure mode as a constraint.

### What these questions test — and what wrong retrieval looks like

| Pair | Failure mode | What a wrong retrieval returns |
|---|---|---|
| 1 (edge detection & negative weights) | **multi-hop** | A chunk about weights/bias in general — which mentions "negative weights" but not the edge-detection contrast mechanism. The answer would be partially correct but miss the *why*. |
| 2 (temperature & derivatives) | **negation / misconception** | A chunk about backpropagation or gradients — which matches the words "derivative" and "temperature" literally but uses them in a different sense. The answer would confidently give the *wrong* definition. |
| 3 (word embedding encodes position + context) | **multi-hop** | A chunk about word embeddings covering only token identity (the lookup table), missing the positional encoding and context-soaking aspects. The answer would be half-right. |
| 4 (automatic vs manual feature extraction) | **contrast** | A chunk about deep learning feature learning in general — which confirms features are learned automatically but doesn't draw the manual-vs-automatic pipeline distinction. The answer would miss the contrast. |
| 5 (ML output from weather inputs) | **multi-hop** | A chunk that only covers the weather scenario without the health-parameter parallel — correct for one use case but missing the generalisation point about what makes the underlying process work. |
