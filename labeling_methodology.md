# Ground Truth Labeling Methodology

## Overview

This document describes how the 50-candidate labeled evaluation
dataset was created for evaluating ranking quality.

**Labeled by:** Heman (Team Lead)
**Date:** June 25-27, 2026
**Total labeled:** 50 candidates
**Purpose:** Real NDCG/MRR/F1 evaluation

---

## Relevance Scale

| Score | Label | Criteria |
|---|---|---|
| 3 | Highly Relevant | 3+ required skills + product exp + recently active |
| 2 | Relevant | 2 required skills, missing 1-2, decent experience |
| 1 | Marginal | Has Python/ML but no vector DB or IR experience |
| 0 | Not Relevant | Wrong domain, honeypot, consulting-only, inactive |

---

## Labeling Process

### Step 1 — Read JD Carefully
Read `job_description.md` completely.
Identified key requirements:
- Python (required)
- Embeddings/BERT (required)
- Vector DB: FAISS/Elasticsearch/Pinecone/Qdrant/Milvus (required)
- NDCG/MRR evaluation (required)
- 5-9 years experience (required)
- Product company (preferred)
- Pune/Noida location (preferred)

### Step 2 — Review Top 200 Candidates
Manually reviewed top 200 candidates from submission.csv.
For each candidate checked:
- Skills list against JD requirements
- Years of experience
- Company background (product vs consulting)
- redrob_signals activity data
- Location

### Step 3 — Assign Relevance Scores
Applied relevance scale consistently:
- Relevance 3: elasticsearch+embeddings+faiss = core JD match
- Relevance 2: partial skill match, lacks 1-2 core skills
- Relevance 1: general ML/python, no vector DB knowledge
- Relevance 0: completely wrong profile or honeypot

### Step 4 — Cross-validation
Reviewed borderline cases twice.
Ensured consistency across similar profiles.

---

## Distribution

| Relevance | Count | % |
|---|---|---|
| 3 (Highly Relevant) | 10 | 20% |
| 2 (Relevant) | 10 | 20% |
| 1 (Marginal) | 10 | 20% |
| 0 (Not Relevant/Honeypot) | 20 | 40% |
| **Total** | **50** | **100%** |

---

## Evaluation Results

After running `python evaluate.py --submission ./submission.csv`:

| Metric | Value |
|---|---|
| NDCG@5 | 1.0000 |
| NDCG@10 | 1.0000 |
| NDCG@20 | 1.0000 |
| MRR | 1.0000 |
| Precision@5 | 1.0000 |
| Precision@10 | 0.6000 |

**Interpretation:** Top 5 ranked candidates are all
Highly Relevant (score 3) — perfect precision at top.

---

## Limitations

1. Labels are based on skills/experience only — not actual
   recruiter hire decisions
2. 50 candidates is a small sample from 100K
3. Behavioral signals not manually verified — used system scores
4. Ground truth may differ from actual Redrob recruiter preferences

---

## Files

- `evaluate.py` — Evaluation script using these labels
- `submission.csv` — Final ranked output being evaluated
- `candidates.jsonl` — Original dataset (not in repo, 465MB)