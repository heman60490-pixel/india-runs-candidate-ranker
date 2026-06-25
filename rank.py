"""
rank.py — India Runs Hackathon Submission Script
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import json
import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────
# REAL JOB DESCRIPTION
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
- Production ML deployment

Good to Have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models
- HR-tech or marketplace product experience
- Distributed systems, large-scale inference optimization
- Open-source contributions in AI/ML

Ideal: 6-8 years, product company, shipped ranking/search to real users at scale.
Located in or willing to relocate to Noida/Pune. Active on job market.
"""

# ─────────────────────────────────────────────────────
# SKILLS
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
    "hr tech", "open source",
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
    "saas", "product company",
}

TOP_K = 100

# ─────────────────────────────────────────────────────
# LOAD
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
# HONEYPOT
# ─────────────────────────────────────────────────────
def is_honeypot(c):
    skills = c.get("skills", [])
    sig    = c.get("redrob_signals", {})
    exp    = c.get("experience", [])

    expert_skills = [s for s in skills if isinstance(s, dict) and
                     s.get("proficiency","").lower() in ("expert","advanced")]
    if len(expert_skills) > 14:
        return True

    completeness = sig.get("profile_completeness", sig.get("completeness_score", None))
    if completeness == 100 and len(exp) == 0:
        return True

    for e in exp:
        if (e.get("duration_months") or 0) > 300:
            return True

    engagement = sig.get("platform_engagement_score", sig.get("engagement_score", None))
    apps = sig.get("applications_submitted", sig.get("applications_count", 0))
    if engagement == 0 and (apps or 0) > 50:
        return True

    title = (c.get("current_title","") or "").lower()
    non_tech = ["marketing manager","sales manager","hr manager",
                "business development","account manager","finance manager"]
    if any(t in title for t in non_tech):
        sk = {(s.get("name","") if isinstance(s,dict) else s).lower() for s in skills}
        if len({"python","machine learning","nlp","embeddings","faiss"} & sk) >= 3:
            return True
    return False

# ─────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────
def extract_text(c):
    parts = []
    title = c.get("current_title","") or c.get("desired_title","")
    if title:
        parts.extend([title]*3)
    for exp in c.get("experience",[]):
        parts.append(exp.get("title",""))
        parts.append((exp.get("description","") or "")[:300])
        parts.append(exp.get("company",""))
    for s in c.get("skills",[]):
        parts.append(s.get("name","") if isinstance(s,dict) else s)
    for edu in c.get("education",[]):
        parts.append(edu.get("degree",""))
        parts.append(edu.get("field",""))
    parts.append((c.get("summary","") or c.get("bio","") or "")[:400])
    return " ".join(filter(None, parts))[:1200]

def extract_skills(c):
    sk = set()
    for s in c.get("skills",[]):
        name = (s.get("name","") if isinstance(s,dict) else s or "")
        sk.add(name.lower().strip())
    return sk

# ─────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────
def skills_score(candidate_skills):
    req_matched = len(JD_REQUIRED_SKILLS & candidate_skills)
    opt_matched = len(JD_OPTIONAL_SKILLS & candidate_skills)
    req_missing = len(JD_REQUIRED_SKILLS) - req_matched
    total_w = len(JD_REQUIRED_SKILLS)*1.0 + len(JD_OPTIONAL_SKILLS)*0.5
    matched_w = req_matched*1.0 + opt_matched*0.5
    penalty = (req_missing / max(len(JD_REQUIRED_SKILLS),1)) * 0.15
    return round(max(0.0, min(1.0, matched_w/total_w - penalty)), 4)

def exp_score(c):
    exp_list = c.get("experience",[])
    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    yrs = total_months/12.0
    if yrs <= 0:
        yrs = float(c.get("years_of_experience",0) or 0)
    if yrs < 1:   return 0.1
    elif yrs < 3: return 0.4
    elif yrs <= 9: return round(min((yrs-3)/6.0*0.7+0.4, 1.0), 4)
    elif yrs <= 12: return 0.8
    else: return 0.65

def beh_score(c):
    sig = c.get("redrob_signals",{})
    if not sig: return 0.25
    scores = []
    for f in ("days_since_last_active","last_active_days","days_inactive"):
        v = sig.get(f)
        if v is not None:
            d = float(v)
            if   d<=7:   scores.append((1.00,2.0))
            elif d<=14:  scores.append((0.85,2.0))
            elif d<=30:  scores.append((0.65,2.0))
            elif d<=90:  scores.append((0.35,2.0))
            elif d<=180: scores.append((0.15,2.0))
            else:        scores.append((0.05,2.0))
            break
    for f in ("profile_completeness","profile_score","completeness_score"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0),1.0)); break
    for f in ("platform_engagement_score","engagement_score","engagement"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0),1.5)); break
    for f in ("job_seeking_intent","open_to_work","actively_looking","is_active"):
        v = sig.get(f)
        if v is not None:
            if isinstance(v,bool): scores.append((1.0 if v else 0.2,1.5))
            elif isinstance(v,(int,float)): scores.append((min(float(v)/100.0,1.0),1.5))
            break
    for f in ("recruiter_response_rate","response_rate"):
        v = sig.get(f)
        if v is not None:
            scores.append((min(float(v)/100.0,1.0),1.0)); break
    if not scores: return 0.25
    tw = sum(w for _,w in scores)
    ws = sum(s*w for s,w in scores)
    return round(ws/tw, 4)

def consulting_penalty(c):
    exp_list = c.get("experience",[])
    if not exp_list: return 0.0
    cc = sum(1 for e in exp_list if any(f in (e.get("company","") or "").lower() for f in CONSULTING_FIRMS))
    ratio = cc/len(exp_list)
    if ratio >= 0.9: return 0.30
    elif ratio >= 0.6: return 0.15
    return 0.0

def product_bonus(c):
    for e in c.get("experience",[]):
        combined = (e.get("company","") or "").lower() + " " + (e.get("description","") or "").lower()
        if any(p in combined for p in PRODUCT_COMPANIES):
            return 0.08
    return 0.0

def location_score(c):
    loc = (c.get("location","") or c.get("city","") or "").lower()
    if any(x in loc for x in ("pune","noida")): return 0.08
    if any(x in loc for x in ("delhi","ncr","hyderabad","mumbai","bangalore","bengaluru")): return 0.04
    if "india" in loc: return 0.02
    return 0.0

def notice_score(c):
    sig = c.get("redrob_signals",{})
    for f in ("notice_period_days","notice_period","notice_days"):
        v = sig.get(f)
        if v is not None:
            d = float(v)
            if d<=0: return 0.10
            elif d<=30: return 0.08
            elif d<=60: return 0.04
            return 0.0
    return 0.05

# ─────────────────────────────────────────────────────
# REASONING
# ─────────────────────────────────────────────────────
def reasoning(c, sk, sem, final, rank):
    title = c.get("current_title","") or c.get("desired_title","")
    exp_list = c.get("experience",[])
    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    yrs = round(total_months/12.0,1)
    if yrs==0: yrs = float(c.get("years_of_experience",0) or 0)
    matched_req = sorted(JD_REQUIRED_SKILLS & sk)[:3]
    missing_req = sorted(JD_REQUIRED_SKILLS - sk)[:2]
    location = c.get("location","") or c.get("city","")
    sig = c.get("redrob_signals",{})
    days_inactive = None
    for f in ("days_since_last_active","last_active_days"):
        v = sig.get(f)
        if v is not None:
            days_inactive = float(v); break

    has_product = any(
        any(p in (e.get("company","") or "").lower() for p in PRODUCT_COMPANIES)
        for e in exp_list
    )

    p1 = []
    if title and yrs>0: p1.append(f"{yrs}-year {title}")
    elif title: p1.append(title)
    elif yrs>0: p1.append(f"{yrs} years experience")
    if matched_req: p1.append(f"strong match on {', '.join(matched_req)}")
    if has_product: p1.append("product company background")
    if location: p1.append(f"based in {location}")
    s1 = (", ".join(p1) or "Candidate") + "."

    concerns = []
    if missing_req: concerns.append(f"missing: {', '.join(missing_req)}")
    if days_inactive and days_inactive>180: concerns.append(f"inactive {int(days_inactive)} days")
    elif days_inactive and days_inactive>90: concerns.append(f"low activity ({int(days_inactive)} days)")
    for e in exp_list:
        if any(f in (e.get("company","") or "").lower() for f in CONSULTING_FIRMS):
            concerns.append("consulting background"); break

    if rank<=10: tone = f"Top-{rank} fit — strong alignment for Senior AI Engineer role"
    elif rank<=25: tone = f"Rank {rank} — good signal alignment, recommended for screening"
    elif rank<=50: tone = f"Rank {rank} — moderate fit, worth reviewing"
    elif rank<=75: tone = f"Rank {rank} — partial fit"
    else: tone = f"Rank {rank} — below threshold"

    s2 = f"{tone}; concerns: {'; '.join(concerns)}." if concerns else f"{tone} — no major gaps."
    return f"{s1} {s2}"

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
    print(f"   Removed {len(candidates)-len(clean)} honeypots — {len(clean):,} remain")
    _, valid_candidates = zip(*clean)
    valid_candidates = list(valid_candidates)

    # 3. Build texts
    print("📝 Building text representations...")
    texts = [extract_text(c) for c in valid_candidates]

    # 4. TF-IDF pre-filter — top 5000 only
    print("⚡ TF-IDF pre-filtering top 5000...")
    jd_keywords = " ".join(JD_REQUIRED_SKILLS | JD_OPTIONAL_SKILLS)
    tfidf = TfidfVectorizer(max_features=10000)
    all_texts = [jd_keywords] + texts
    tfidf_matrix = tfidf.fit_transform(all_texts)
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    tfidf_scores = cos_sim(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    top_5000 = np.argsort(tfidf_scores)[::-1][:5000]
    valid_candidates = [valid_candidates[i] for i in top_5000]
    texts = [texts[i] for i in top_5000]
    print(f"   Pre-filtered to {len(texts):,} candidates in {time.time()-t0:.1f}s")

    # 5. BERT encode only top 5000
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("⚡ Encoding JD...")
    jd_vec = model.encode([JOB_DESCRIPTION], show_progress_bar=False)
    print(f"⚡ BERT encoding {len(texts):,} candidates...")
    cand_vecs = model.encode(texts, batch_size=512, show_progress_bar=True, convert_to_numpy=True)
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

    sk_scores = np.array(sk_scores)
    ex_scores = np.array(ex_scores)
    bh_scores = np.array(bh_scores)
    loc_scores = np.array(loc_scores)
    cp_scores = np.array(cp_scores)
    pb_scores = np.array(pb_scores)
    ns_scores = np.array(ns_scores)

    # 7. Final score
    final_scores = np.clip(
        sem_scores*0.35 + sk_scores*0.30 + ex_scores*0.15 + bh_scores*0.20
        + loc_scores + pb_scores + ns_scores - cp_scores,
        0.0, 1.5
    )

    # 8. Top 100
    top_idx = np.argsort(final_scores)[::-1][:TOP_K]

    # 9. Build output
    print("✍️  Generating reasoning...")
    rows = []
    for rank_pos, idx in enumerate(top_idx, 1):
        c = valid_candidates[idx]
        cid = c.get("candidate_id", c.get("id", f"CAND_{idx:07d}"))
        fs = round(float(final_scores[idx]),4)
        ss = round(float(sem_scores[idx]),4)
        r = reasoning(c, skill_sets[idx], ss, fs, rank_pos)
        rows.append({"candidate_id":cid,"rank":rank_pos,"score":fs,"reasoning":r})

    for i in range(1,len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    df = pd.DataFrame(rows, columns=["candidate_id","rank","score","reasoning"])
    df.to_csv(output_path, index=False, encoding="utf-8")

    elapsed = time.time()-t0
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"📄 Saved: {output_path}")
    print(f"🔢 Score range: {df['score'].max():.4f} → {df['score'].min():.4f}")
    print(f"\n📊 Evaluation:")
    scores = df['score'].values
    print(f"   Std dev: {scores.std():.4f}")
    print(f"   Top-10 avg: {scores[:10].mean():.4f}")
    print(f"   Spread: {scores[0]-scores[-1]:.4f}")
    print("\nTop 5:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)