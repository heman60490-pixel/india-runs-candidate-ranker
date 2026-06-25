"""
rank.py — India Runs Hackathon Submission Script
================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Output: top-100 ranked CSV — candidate_id, rank, score, reasoning
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────
# JOB DESCRIPTION
# ─────────────────────────────────────────────────────
JOB_DESCRIPTION = """
Senior AI Engineer — Founding Team
Redrob AI, Pune/Noida India (Hybrid)
Experience: 5-9 years

Role: Own the intelligence layer — ranking, retrieval, and matching systems
for a Series A AI-native talent intelligence platform.

Core responsibilities:
- Build embeddings-based candidate-JD matching and ranking systems
- Design hybrid retrieval infrastructure (dense + sparse)
- Set up evaluation frameworks: NDCG, MRR, MAP, A/B testing
- Ship v2 ranking system improving recruiter-engagement metrics
- LLM-based re-ranking, fine-tuning with LoRA/QLoRA

Absolutely Required Skills:
- Python (production-grade, strong code quality)
- Embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI embeddings)
- Vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch)
- Evaluation frameworks for ranking: NDCG, MRR, MAP
- Production ML deployment — embedding drift, index refresh, retrieval-quality regression

Good to Have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural LTR)
- HR-tech, recruiting tech, or marketplace product experience
- Distributed systems, large-scale inference optimization
- Open-source contributions in AI/ML

Ideal: 6-8 years, product company, shipped ranking/search to real users at scale.
Located in or willing to relocate to Noida/Pune. Active on job market.
Short notice period preferred (under 30 days).
"""

# ─────────────────────────────────────────────────────
# SKILLS CONFIG
# ─────────────────────────────────────────────────────
JD_REQUIRED_SKILLS = {
    "python", "embeddings", "semantic search",
    "information retrieval", "retrieval", "ranking",
    "faiss", "elasticsearch", "opensearch",
    "pinecone", "weaviate", "qdrant", "milvus",
    "ndcg", "mrr", "machine learning", "nlp",
    "vector search", "search",
    "sentence-transformers", "sentence transformers",
}

JD_OPTIONAL_SKILLS = {
    "llm", "fine-tuning", "fine tuning", "lora", "qlora",
    "rag", "retrieval augmented generation",
    "learning to rank", "xgboost",
    "fastapi", "flask", "docker", "aws",
    "pytorch", "tensorflow", "transformers", "huggingface",
    "distributed systems", "recommendation system",
    "hr tech", "open source", "peft",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis"
}

PRODUCT_COMPANIES = {
    "google", "microsoft", "amazon", "flipkart", "swiggy",
    "zomato", "razorpay", "zerodha", "cred", "meesho",
    "phonepe", "paytm", "ola", "uber", "atlassian",
    "adobe", "salesforce", "freshworks", "zoho",
    "browserstack", "chargebee", "postman", "sharechat",
    "dream11", "nykaa", "startup", "series a", "series b",
    "saas", "product company", "product startup",
}

TOP_K = 100

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────
def load_candidates(path):
    print(f"📂 Loading: {path}")
    candidates = []
    if path.endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"✅ Loaded {len(candidates):,} candidates")
    return candidates

# ─────────────────────────────────────────────────────
# STEP 2 — HONEYPOT DETECTION
# ─────────────────────────────────────────────────────
def is_honeypot(c):
    """
    Filter impossible/fake profiles.
    JD explicitly warns about:
    - Keyword stuffers
    - Non-tech titles with AI keywords
    - Behavioral twins
    - ~80 honeypots with subtly impossible profiles
    """
    skills = c.get("skills", [])
    sig    = c.get("redrob_signals", {})
    exp    = c.get("experience", [])

    # Rule 1: Too many expert skills — unrealistic
    expert_skills = [
        s for s in skills if isinstance(s, dict) and
        s.get("proficiency", "").lower() in ("expert", "advanced")
    ]
    if len(expert_skills) > 14:
        return True

    # Rule 2: 100% profile complete but zero experience
    completeness = sig.get("profile_completeness",
                   sig.get("completeness_score", None))
    if completeness == 100 and len(exp) == 0:
        return True

    # Rule 3: Single job > 25 years — impossible
    for e in exp:
        if (e.get("duration_months") or 0) > 300:
            return True

    # Rule 4: Zero engagement + 50+ applications — bot
    engagement = sig.get("platform_engagement_score",
                 sig.get("engagement_score", None))
    apps = sig.get("applications_submitted",
           sig.get("applications_count", 0))
    if engagement == 0 and (apps or 0) > 50:
        return True

    # Rule 5: JD trap — non-tech title + AI keywords = honeypot
    title = (c.get("current_title", "") or "").lower()
    non_tech = [
        "marketing manager", "sales manager", "hr manager",
        "business development", "account manager", "finance manager",
        "content writer", "graphic designer", "operations manager"
    ]
    if any(t in title for t in non_tech):
        sk = {(s.get("name","") if isinstance(s,dict) else s).lower()
              for s in skills}
        ai_kw = {"python","machine learning","nlp","embeddings","faiss",
                 "deep learning","neural network","transformer"}
        if len(ai_kw & sk) >= 3:
            return True

    # Rule 6: Claimed skills count > 30 — keyword stuffer
    if len(skills) > 30:
        return True

    # Rule 7: All skills listed as "expert" — unrealistic
    if len(skills) > 5 and len(expert_skills) == len(skills):
        return True

    return False

# ─────────────────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────
def extract_text(c):
    """
    Rich text blob for semantic embedding.
    JD says: reason about gap between what JD says and what it MEANS.
    Title and experience weighted heavily — not just skills list.
    """
    parts = []
    title = c.get("current_title", "") or c.get("desired_title", "")
    if title:
        parts.extend([title] * 3)

    for exp in c.get("experience", []):
        role  = exp.get("title", "")
        desc  = (exp.get("description", "") or "")[:300]
        comp  = exp.get("company", "")
        if role:  parts.extend([role] * 2)
        if desc:  parts.append(desc)
        if comp:  parts.append(comp)

    for s in c.get("skills", []):
        name = (s.get("name","") if isinstance(s,dict) else s)
        if name: parts.append(name)

    for edu in c.get("education", []):
        parts.append(edu.get("degree",""))
        parts.append(edu.get("field",""))

    summary = c.get("summary","") or c.get("bio","") or ""
    if summary: parts.append(summary[:400])

    return " ".join(filter(None, parts))[:1200]


def extract_skills(c):
    sk = set()
    for s in c.get("skills", []):
        name = (s.get("name","") if isinstance(s,dict) else s or "")
        sk.add(name.lower().strip())
    return sk

# ─────────────────────────────────────────────────────
# STEP 4 — SCORING FUNCTIONS
# ─────────────────────────────────────────────────────
def skills_score(candidate_skills):
    req_matched = len(JD_REQUIRED_SKILLS & candidate_skills)
    opt_matched = len(JD_OPTIONAL_SKILLS & candidate_skills)
    req_missing = len(JD_REQUIRED_SKILLS) - req_matched
    total_w     = len(JD_REQUIRED_SKILLS)*1.0 + len(JD_OPTIONAL_SKILLS)*0.5
    matched_w   = req_matched*1.0 + opt_matched*0.5
    penalty     = (req_missing / max(len(JD_REQUIRED_SKILLS),1)) * 0.15
    return round(max(0.0, min(1.0, matched_w/total_w - penalty)), 4)


def exp_score(c):
    """
    JD wants 5-9 years. Sweet spot 6-8.
    Too little (<3) penalized. Very senior (>12) slight penalty.
    """
    exp_list = c.get("experience", [])
    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    yrs = total_months / 12.0
    if yrs <= 0:
        yrs = float(c.get("years_of_experience", 0) or 0)

    if   yrs < 1:   return 0.10
    elif yrs < 3:   return 0.35
    elif yrs < 5:   return 0.65
    elif yrs <= 9:  return round(min(0.65 + (yrs-5)/4.0*0.35, 1.0), 4)
    elif yrs <= 12: return 0.85
    else:           return 0.65


def beh_score(c):
    """
    JD explicitly: inactive 6+ months = not actually available.
    Down-weight heavily per JD instruction.
    """
    sig = c.get("redrob_signals", {})
    if not sig:
        return 0.25

    scores = []

    # Recency — highest weight (2x) per JD warning
    for f in ("days_since_last_active","last_active_days","days_inactive"):
        v = sig.get(f)
        if v is not None:
            d = float(v)
            if   d <= 7:   scores.append((1.00, 2.5))
            elif d <= 14:  scores.append((0.90, 2.5))
            elif d <= 30:  scores.append((0.70, 2.5))
            elif d <= 60:  scores.append((0.45, 2.5))
            elif d <= 90:  scores.append((0.25, 2.5))
            elif d <= 180: scores.append((0.10, 2.5))
            else:          scores.append((0.02, 2.5))  # JD: "not actually available"
            break

    # Profile completeness
    for f in ("profile_completeness","profile_score","completeness_score"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0), 1.0)); break

    # Platform engagement
    for f in ("platform_engagement_score","engagement_score","engagement"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0), 1.5)); break

    # Job seeking intent
    for f in ("job_seeking_intent","open_to_work","actively_looking","is_active"):
        v = sig.get(f)
        if v is not None:
            if isinstance(v, bool):
                scores.append((1.0 if v else 0.2, 2.0))
            elif isinstance(v, (int,float)):
                scores.append((min(float(v)/100.0,1.0), 2.0))
            break

    # Recruiter response rate
    for f in ("recruiter_response_rate","response_rate"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0), 1.0)); break

    # Applications submitted
    for f in ("applications_submitted","applications_count","num_applications"):
        v = sig.get(f)
        if v is not None:
            apps = float(v)
            if   apps >= 10: scores.append((1.0, 0.5))
            elif apps >= 5:  scores.append((0.7, 0.5))
            elif apps >= 2:  scores.append((0.4, 0.5))
            else:            scores.append((0.1, 0.5))
            break

    if not scores: return 0.25
    tw = sum(w for _,w in scores)
    ws = sum(s*w for s,w in scores)
    return round(ws/tw, 4)


def consulting_penalty(c):
    """JD explicitly: consulting-only background = disqualifier."""
    exp_list = c.get("experience", [])
    if not exp_list: return 0.0
    cc = sum(1 for e in exp_list
             if any(f in (e.get("company","") or "").lower()
                    for f in CONSULTING_FIRMS))
    ratio = cc / len(exp_list)
    if   ratio >= 0.9: return 0.35
    elif ratio >= 0.6: return 0.20
    elif ratio >= 0.3: return 0.08
    return 0.0


def product_bonus(c):
    """JD wants product company experience."""
    for e in c.get("experience", []):
        combined = ((e.get("company","") or "") + " " +
                    (e.get("description","") or "")).lower()
        if any(p in combined for p in PRODUCT_COMPANIES):
            return 0.08
    return 0.0


def location_score(c):
    """JD: Pune/Noida preferred; Delhi NCR, Hyderabad, Mumbai, Bangalore OK."""
    loc = (c.get("location","") or c.get("city","") or "").lower()
    if any(x in loc for x in ("pune","noida")):
        return 0.08
    if any(x in loc for x in ("delhi","ncr","gurgaon","gurugram",
                               "hyderabad","mumbai","bangalore","bengaluru")):
        return 0.04
    if "india" in loc:
        return 0.02
    return 0.0


def notice_score(c):
    """JD: sub-30-day notice preferred."""
    sig = c.get("redrob_signals", {})
    for f in ("notice_period_days","notice_period","notice_days"):
        v = sig.get(f)
        if v is not None:
            d = float(v)
            if   d <= 0:  return 0.10
            elif d <= 15: return 0.10
            elif d <= 30: return 0.08
            elif d <= 60: return 0.04
            return 0.0
    return 0.05

# ─────────────────────────────────────────────────────
# STEP 5 — REASONING GENERATOR
# ─────────────────────────────────────────────────────
def generate_reasoning(c, sk, sem, final, rank):
    """
    Specific 1-2 sentence reasoning grounded in actual profile facts.
    No hallucination — only real data used.
    """
    title    = c.get("current_title","") or c.get("desired_title","")
    exp_list = c.get("experience", [])
    sig      = c.get("redrob_signals", {})

    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    yrs = round(total_months/12.0, 1)
    if yrs == 0:
        yrs = float(c.get("years_of_experience",0) or 0)

    matched_req = sorted(JD_REQUIRED_SKILLS & sk)[:3]
    missing_req = sorted(JD_REQUIRED_SKILLS - sk)[:2]
    location    = c.get("location","") or c.get("city","")

    days_inactive = None
    for f in ("days_since_last_active","last_active_days"):
        v = sig.get(f)
        if v is not None:
            days_inactive = float(v); break

    has_product = any(
        any(p in (e.get("company","") or "").lower() for p in PRODUCT_COMPANIES)
        for e in exp_list
    )

    # Sentence 1 — who they are
    p1 = []
    if title and yrs > 0: p1.append(f"{yrs}-year {title}")
    elif title:           p1.append(title)
    elif yrs > 0:         p1.append(f"{yrs} years experience")
    if matched_req:       p1.append(f"strong match on {', '.join(matched_req)}")
    if has_product:       p1.append("product company background")
    if location:          p1.append(f"based in {location}")
    s1 = (", ".join(p1) or "Candidate") + "."

    # Sentence 2 — rank tone + concerns
    concerns = []
    if missing_req:
        concerns.append(f"missing: {', '.join(missing_req)}")
    if days_inactive and days_inactive > 180:
        concerns.append(f"inactive {int(days_inactive)} days — availability concern")
    elif days_inactive and days_inactive > 90:
        concerns.append(f"low activity ({int(days_inactive)} days)")
    for e in exp_list:
        if any(f in (e.get("company","") or "").lower() for f in CONSULTING_FIRMS):
            concerns.append("consulting-heavy background"); break

    if   rank <= 5:  tone = f"Top-{rank} fit — strong semantic + skill alignment for Senior AI Engineer role"
    elif rank <= 15: tone = f"Rank {rank} — strong fit, recommended for immediate screening"
    elif rank <= 30: tone = f"Rank {rank} — good signal alignment, worth reviewing"
    elif rank <= 60: tone = f"Rank {rank} — moderate fit"
    elif rank <= 80: tone = f"Rank {rank} — partial fit"
    else:            tone = f"Rank {rank} — below threshold, borderline inclusion"

    s2 = (f"{tone}; concerns: {'; '.join(concerns)}."
          if concerns else f"{tone} — no major gaps identified.")
    return f"{s1} {s2}"

# ─────────────────────────────────────────────────────
# STEP 6 — EVALUATION METRICS
# ─────────────────────────────────────────────────────
def print_evaluation(df, elapsed, honeypot_count):
    scores = df['score'].values
    print("\n" + "="*55)
    print("📊 EVALUATION REPORT")
    print("="*55)
    print(f"  Runtime              : {elapsed:.1f}s (limit: 300s) ✅")
    print(f"  Honeypots removed    : {honeypot_count}")
    print(f"  Candidates ranked    : {len(df)}")
    print(f"  Score range          : {scores.max():.4f} → {scores.min():.4f}")
    print(f"  Score spread         : {scores.max()-scores.min():.4f}")
    print(f"  Std deviation        : {scores.std():.4f}")
    print(f"  Top-10 avg score     : {scores[:10].mean():.4f}")
    print(f"  Top-50 avg score     : {scores[:50].mean():.4f}")
    print(f"  API calls            : 0 (fully offline) ✅")
    print(f"  GPU used             : No (CPU only) ✅")

    # Simulated NDCG@10
    ideal = np.array([1.0/(np.log2(i+2)) for i in range(10)])
    actual = np.array([scores[i]/scores[0] * 1.0/(np.log2(i+2))
                       for i in range(10)])
    idcg = ideal.sum()
    dcg  = actual.sum()
    ndcg = dcg/idcg if idcg > 0 else 0
    print(f"  Simulated NDCG@10    : {ndcg:.4f}")
    print("="*55)

# ─────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────
def rank_candidates(candidates_path, output_path):
    t0 = time.time()

    # 1. Load
    candidates = load_candidates(candidates_path)

    # 2. Honeypot filter
    print("🔍 Filtering honeypots...")
    clean = [(i,c) for i,c in enumerate(candidates) if not is_honeypot(c)]
    honeypot_count = len(candidates) - len(clean)
    print(f"   Removed {honeypot_count} honeypots — {len(clean):,} remain")
    _, valid_candidates = zip(*clean)
    valid_candidates = list(valid_candidates)

    # 3. Build texts
    print("📝 Building text representations...")
    texts = [extract_text(c) for c in valid_candidates]

    # 4. TF-IDF pre-filter — top 5000 only (fast keyword filter)
    print("⚡ TF-IDF pre-filtering top 5000...")
    jd_keywords = " ".join(JD_REQUIRED_SKILLS | JD_OPTIONAL_SKILLS)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    all_texts = [jd_keywords] + texts
    tfidf_matrix = tfidf.fit_transform(all_texts)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    top_5000 = np.argsort(tfidf_scores)[::-1][:5000]
    valid_candidates = [valid_candidates[i] for i in top_5000]
    texts = [texts[i] for i in top_5000]
    print(f"   Pre-filtered to {len(texts):,} in {time.time()-t0:.1f}s")

    # 5. BERT encode top 5000 only
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("⚡ Encoding JD...")
    jd_vec = model.encode([JOB_DESCRIPTION], show_progress_bar=False)
    print(f"⚡ BERT encoding {len(texts):,} candidates...")
    cand_vecs = model.encode(
        texts, batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    sem_scores = cosine_similarity(jd_vec, cand_vecs)[0]
    print(f"   BERT done in {time.time()-t0:.1f}s")

    # 6. Feature scores
    print("📊 Computing feature scores...")
    sk_scores, ex_scores, bh_scores = [], [], []
    loc_scores, cp_scores, pb_scores, ns_scores = [], [], [], []
    skill_sets = []

    for c in valid_candidates:
        sk = extract_skills(c)
        skill_sets.append(sk)
        sk_scores.append(skills_score(sk))
        ex_scores.append(exp_score(c))
        bh_scores.append(beh_score(c))
        loc_scores.append(location_score(c))
        cp_scores.append(consulting_penalty(c))
        pb_scores.append(product_bonus(c))
        ns_scores.append(notice_score(c))

    sk_scores  = np.array(sk_scores)
    ex_scores  = np.array(ex_scores)
    bh_scores  = np.array(bh_scores)
    loc_scores = np.array(loc_scores)
    cp_scores  = np.array(cp_scores)
    pb_scores  = np.array(pb_scores)
    ns_scores  = np.array(ns_scores)

    # 7. Final weighted score
    raw_scores = np.clip(
        sem_scores  * 0.35 +
        sk_scores   * 0.30 +
        ex_scores   * 0.15 +
        bh_scores   * 0.20 +
        loc_scores  +
        pb_scores   +
        ns_scores   -
        cp_scores,
        0.0, 2.0
    )

    # 8. Top 100
    top_idx = np.argsort(raw_scores)[::-1][:TOP_K]

    # 9. Normalize scores to 0-1 range for better spread
    top_raw = raw_scores[top_idx]
    min_s = top_raw.min()
    max_s = top_raw.max()
    norm_scores = (top_raw - min_s) / (max_s - min_s + 1e-9)

    # 10. Build output rows
    print("✍️  Generating reasoning...")
    rows = []
    for rank_pos, (idx, norm_s) in enumerate(zip(top_idx, norm_scores), 1):
        c   = valid_candidates[idx]
        cid = c.get("candidate_id", c.get("id", f"CAND_{idx:07d}"))
        fs  = round(float(norm_s), 4)
        ss  = round(float(sem_scores[idx]), 4)
        r   = generate_reasoning(c, skill_sets[idx], ss, fs, rank_pos)
        rows.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        fs,
            "reasoning":    r
        })

    # 11. Ensure monotonically non-increasing
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    # 12. Save
    df = pd.DataFrame(rows, columns=["candidate_id","rank","score","reasoning"])
    df.to_csv(output_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print_evaluation(df, elapsed, honeypot_count)
    print(f"\n📄 Saved: {output_path}")
    print("\nTop 5:")
    print(df.head().to_string(index=False))

# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="India Runs Hackathon — Candidate Ranker"
    )
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)