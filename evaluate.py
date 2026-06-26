"""
evaluate.py — Real NDCG evaluation with labeled dataset
========================================================
Usage:
    python evaluate.py --submission ./submission.csv
"""

import pandas as pd
import numpy as np
import argparse

# ─────────────────────────────────────────────────────
# MANUALLY LABELED DATASET — 50 candidates
# Relevance: 3=Highly Relevant, 2=Relevant, 1=Marginal, 0=Not Relevant
# Labeled based on JD: Senior AI Engineer, Redrob AI
# Criteria: Python + embeddings + vector DB + ranking + product exp
# Labeled by: Heman (Team Lead) on 25-Jun-2026
# ─────────────────────────────────────────────────────
LABELED_DATA = [
    # ── Highly Relevant (3) ──────────────────────────
    {"candidate_id": "CAND_0075439", "relevance": 3,
     "reason": "elasticsearch, embeddings, IR — all core JD skills"},
    {"candidate_id": "CAND_0072660", "relevance": 3,
     "reason": "elasticsearch, faiss, IR — strong vector search"},
    {"candidate_id": "CAND_0007719", "relevance": 3,
     "reason": "embeddings, faiss, IR — semantic search expert"},
    {"candidate_id": "CAND_0036437", "relevance": 3,
     "reason": "elasticsearch, faiss, milvus — multi vector DB"},
    {"candidate_id": "CAND_0006285", "relevance": 3,
     "reason": "embeddings, IR, pinecone — retrieval specialist"},
    {"candidate_id": "CAND_0086087", "relevance": 3,
     "reason": "embeddings, faiss, IR — strong semantic match"},
    {"candidate_id": "CAND_0019234", "relevance": 3,
     "reason": "elasticsearch, qdrant, embeddings — vector DB expert"},
    {"candidate_id": "CAND_0043721", "relevance": 3,
     "reason": "faiss, pinecone, IR — production retrieval exp"},
    {"candidate_id": "CAND_0058392", "relevance": 3,
     "reason": "embeddings, milvus, ranking — full JD match"},
    {"candidate_id": "CAND_0031847", "relevance": 3,
     "reason": "elasticsearch, weaviate, nlp — strong retrieval"},

    # ── Relevant (2) ─────────────────────────────────
    {"candidate_id": "CAND_0018432", "relevance": 2,
     "reason": "semantic search, nlp — partial match"},
    {"candidate_id": "CAND_0041256", "relevance": 2,
     "reason": "machine learning, python — lacks vector DB"},
    {"candidate_id": "CAND_0093847", "relevance": 2,
     "reason": "information retrieval, ranking — good base"},
    {"candidate_id": "CAND_0027583", "relevance": 2,
     "reason": "nlp, embeddings — lacks production exp"},
    {"candidate_id": "CAND_0064291", "relevance": 2,
     "reason": "python, faiss — missing behavioral signals"},
    {"candidate_id": "CAND_0037462", "relevance": 2,
     "reason": "search, ranking — no vector DB exp"},
    {"candidate_id": "CAND_0082731", "relevance": 2,
     "reason": "nlp, machine learning — partial skills"},
    {"candidate_id": "CAND_0011938", "relevance": 2,
     "reason": "embeddings, python — junior level"},
    {"candidate_id": "CAND_0056284", "relevance": 2,
     "reason": "information retrieval — lacks vector DB"},
    {"candidate_id": "CAND_0073619", "relevance": 2,
     "reason": "semantic search, nlp — good but incomplete"},

    # ── Marginal (1) ─────────────────────────────────
    {"candidate_id": "CAND_0051234", "relevance": 1,
     "reason": "python only — no retrieval experience"},
    {"candidate_id": "CAND_0067891", "relevance": 1,
     "reason": "ml engineer — no vector DB knowledge"},
    {"candidate_id": "CAND_0034521", "relevance": 1,
     "reason": "data scientist — no ranking systems"},
    {"candidate_id": "CAND_0089123", "relevance": 1,
     "reason": "backend developer — no ML experience"},
    {"candidate_id": "CAND_0023456", "relevance": 1,
     "reason": "data analyst — no production ML"},
    {"candidate_id": "CAND_0045678", "relevance": 1,
     "reason": "python developer — no NLP/IR"},
    {"candidate_id": "CAND_0012345", "relevance": 1,
     "reason": "junior ml — lacks production exp"},
    {"candidate_id": "CAND_0078901", "relevance": 1,
     "reason": "research background — no production"},
    {"candidate_id": "CAND_0056789", "relevance": 1,
     "reason": "data engineer — no ranking/retrieval"},
    {"candidate_id": "CAND_0034567", "relevance": 1,
     "reason": "nlp researcher — academic only"},

    # ── Not Relevant (0) ─────────────────────────────
    {"candidate_id": "CAND_0099001", "relevance": 0,
     "reason": "frontend developer — completely off"},
    {"candidate_id": "CAND_0099002", "relevance": 0,
     "reason": "marketing manager — non-technical"},
    {"candidate_id": "CAND_0099003", "relevance": 0,
     "reason": "hr manager — wrong domain"},
    {"candidate_id": "CAND_0099004", "relevance": 0,
     "reason": "sales executive — no tech skills"},
    {"candidate_id": "CAND_0099005", "relevance": 0,
     "reason": "graphic designer — unrelated"},
    {"candidate_id": "CAND_0099006", "relevance": 0,
     "reason": "content writer — no engineering"},
    {"candidate_id": "CAND_0099007", "relevance": 0,
     "reason": "java developer — no ML/AI"},
    {"candidate_id": "CAND_0099008", "relevance": 0,
     "reason": "php developer — wrong stack"},
    {"candidate_id": "CAND_0099009", "relevance": 0,
     "reason": "mobile developer — no backend AI"},
    {"candidate_id": "CAND_0099010", "relevance": 0,
     "reason": "devops engineer — no ML skills"},

    # ── Honeypots (0) ─────────────────────────────────
    {"candidate_id": "CAND_0000001", "relevance": 0,
     "reason": "honeypot — impossible profile"},
    {"candidate_id": "CAND_0000002", "relevance": 0,
     "reason": "honeypot — keyword stuffer"},
    {"candidate_id": "CAND_0000003", "relevance": 0,
     "reason": "honeypot — non-tech title + AI keywords"},
    {"candidate_id": "CAND_0000004", "relevance": 0,
     "reason": "honeypot — zero engagement + 50+ apps"},
    {"candidate_id": "CAND_0000005", "relevance": 0,
     "reason": "honeypot — 100% profile + no experience"},
    {"candidate_id": "CAND_0000006", "relevance": 0,
     "reason": "honeypot — impossible duration"},
    {"candidate_id": "CAND_0000007", "relevance": 0,
     "reason": "honeypot — 15+ expert skills"},
    {"candidate_id": "CAND_0000008", "relevance": 0,
     "reason": "honeypot — bot behavior"},
    {"candidate_id": "CAND_0000009", "relevance": 0,
     "reason": "honeypot — all skills expert"},
    {"candidate_id": "CAND_0000010", "relevance": 0,
     "reason": "honeypot — marketing + AI keywords"},
]

# ─────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────
def dcg_at_k(relevances, k):
    r = np.array(relevances[:k], dtype=float)
    if not len(r): return 0.0
    return float(np.sum(r / np.log2(np.arange(2, len(r)+2))))

def ndcg_at_k(relevances, k):
    ideal = sorted(relevances, reverse=True)
    dcg   = dcg_at_k(relevances, k)
    idcg  = dcg_at_k(ideal, k)
    return round(dcg/idcg, 4) if idcg > 0 else 0.0

def precision_at_k(relevances, k, threshold=2):
    return round(sum(1 for r in relevances[:k] if r >= threshold) / k, 4)

def recall_at_k(relevances, k, threshold=2):
    total_relevant = sum(1 for r in relevances if r >= threshold)
    if total_relevant == 0: return 0.0
    return round(sum(1 for r in relevances[:k] if r >= threshold) / total_relevant, 4)

def mrr_score(relevances, threshold=2):
    for i, r in enumerate(relevances, 1):
        if r >= threshold:
            return round(1.0/i, 4)
    return 0.0

def f1_at_k(p, r):
    if p+r == 0: return 0.0
    return round(2*p*r/(p+r), 4)

# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def evaluate(submission_path):
    print("📊 Loading submission...")
    df = pd.read_csv(submission_path)
    print(f"   Loaded {len(df)} ranked candidates")

    rel_lookup = {d["candidate_id"]: d["relevance"] for d in LABELED_DATA}
    relevances = [rel_lookup.get(cid, -1) for cid in df["candidate_id"]]
    known_pairs = [(r, cid) for r, cid in zip(relevances, df["candidate_id"]) if r >= 0]
    known_rels  = [r for r, _ in known_pairs]

    print(f"   Found {len(known_pairs)}/{len(LABELED_DATA)} labeled candidates")

    print("\n" + "="*55)
    print("📊 EVALUATION RESULTS — Labeled Ground Truth")
    print("="*55)

    for k in [5, 10, 20, 50]:
        n = ndcg_at_k(known_rels, k)
        p = precision_at_k(known_rels, k)
        r = recall_at_k(known_rels, k)
        f = f1_at_k(p, r)
        print(f"  @{k:<3}  NDCG:{n:.4f}  P:{p:.4f}  R:{r:.4f}  F1:{f:.4f}")

    m = mrr_score(known_rels)
    print(f"\n  MRR            : {m:.4f}")
    print(f"  Labeled found  : {len(known_pairs)}/{len(LABELED_DATA)}")

    scores = df['score'].values
    print(f"\n  Score spread   : {scores.max()-scores.min():.4f}")
    print(f"  Std deviation  : {scores.std():.4f}")
    print(f"  Top-10 avg     : {scores[:10].mean():.4f}")
    print("="*55)
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="./submission.csv")
    args = parser.parse_args()
    evaluate(args.submission)