"""
rank.py — India Runs Hackathon Submission Script
================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Produces top-100 ranked CSV with columns:
    candidate_id, rank, score, reasoning
"""

import argparse
import json
import re
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────
# REAL JOB DESCRIPTION — Senior AI Engineer, Redrob AI
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
- Evaluation frameworks for ranking: NDCG, MRR, MAP, offline-to-online correlation
- Production ML deployment — embedding drift, index refresh, retrieval-quality regression

Good to Have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural LTR)
- HR-tech, recruiting tech, or marketplace product experience
- Distributed systems, large-scale inference optimization
- Open-source contributions in AI/ML

Disqualifiers (explicitly stated):
- Pure research / academic only, no production deployment
- Only LangChain/OpenAI wrapper experience under 12 months
- Consulting firms only (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) without product company experience
- Computer vision / speech / robotics without NLP/IR experience
- No production code written in last 18 months

Ideal profile:
- 6-8 years total, 4-5 years applied ML at product companies
- Shipped end-to-end ranking/search/recommendation to real users at scale
- Strong opinions on retrieval, evaluation, LLM integration
- Located in or willing to relocate to Noida/Pune
- Active on job market (recent platform activity)
- Short notice period preferred (under 30 days)
"""

# ─────────────────────────────────────────────────────
# REQUIRED & OPTIONAL SKILLS (from JD above)
# ─────────────────────────────────────────────────────
JD_REQUIRED_SKILLS = {
    # Core must-haves
    "python",
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "vector search", "semantic search",
    "information retrieval", "retrieval",
    "ranking", "ranking systems", "candidate ranking",
    "faiss", "elasticsearch", "opensearch",
    "pinecone", "weaviate", "qdrant", "milvus",
    "ndcg", "map", "mrr", "evaluation",
    "machine learning", "nlp",
    "production ml", "ml engineering",
    "bge", "e5",
    "hybrid search", "hybrid retrieval",
    "recommendation system", "search",
}

JD_OPTIONAL_SKILLS = {
    # Nice to have
    "llm", "large language model",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "learning to rank", "ltr", "xgboost",
    "distributed systems", "large scale inference",
    "open source", "fastapi", "flask",
    "docker", "kubernetes", "aws",
    "transformers", "huggingface", "pytorch", "tensorflow",
    "rag", "retrieval augmented generation",
    "hr tech", "recruiting", "talent acquisition",
}

# Consulting firms = disqualifier signal
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis"
}

# Product companies = strong positive signal
PRODUCT_COMPANIES = {
    "google", "microsoft", "amazon", "flipkart", "swiggy",
    "zomato", "razorpay", "zerodha", "cred", "meesho",
    "phonepe", "paytm", "ola", "uber", "atlassian",
    "adobe", "salesforce", "oracle", "intuit", "freshworks",
    "zoho", "browserstack", "chargebee", "postman",
    "sharechat", "dream11", "moj", "licious", "nykaa",
    "startup", "series a", "series b", "series c",
    "saas", "product company", "product startup",
}

TOP_K = 100

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD CANDIDATES
# ─────────────────────────────────────────────────────
def load_candidates(path: str) -> list:
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
def is_honeypot(c: dict) -> bool:
    """
    Filter impossible/fake profiles.
    JD explicitly warns: keyword-stuffers and impossible profiles exist.
    """
    skills  = c.get("skills", [])
    sig     = c.get("redrob_signals", {})
    exp     = c.get("experience", [])

    # Too many "expert" skills — unrealistic
    expert_skills = [
        s for s in skills
        if isinstance(s, dict) and
        s.get("proficiency", "").lower() in ("expert", "advanced")
    ]
    if len(expert_skills) > 14:
        return True

    # Profile 100% complete but zero experience
    completeness = sig.get("profile_completeness",
                   sig.get("completeness_score", None))
    if completeness == 100 and len(exp) == 0:
        return True

    # Single job duration > 25 years — impossible
    for e in exp:
        if (e.get("duration_months") or 0) > 300:
            return True

    # Zero engagement but 50+ applications — bot
    engagement = sig.get("platform_engagement_score",
                 sig.get("engagement_score", None))
    apps = sig.get("applications_submitted",
           sig.get("applications_count", 0))
    if engagement == 0 and (apps or 0) > 50:
        return True

    # JD explicitly warns: "Marketing Manager" with AI keywords = trap
    title = (c.get("current_title", "") or "").lower()
    non_tech_titles = [
        "marketing manager", "sales manager", "hr manager",
        "business development", "account manager", "finance manager"
    ]
    if any(t in title for t in non_tech_titles):
        skill_names = {
            (s.get("name","") if isinstance(s,dict) else s).lower()
            for s in skills
        }
        # Has AI keywords but non-tech title = honeypot
        ai_keywords = {"python","machine learning","nlp","embeddings","faiss"}
        if len(ai_keywords & skill_names) >= 3:
            return True

    return False

# ─────────────────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────
def extract_text_for_embedding(c: dict) -> str:
    """
    Rich text blob for semantic embedding.
    JD says: reason about gap between what JD says and what it MEANS.
    Weight title and experience heavily — not just skills list.
    """
    parts = []

    # Title — most important signal (3x weight)
    title = c.get("current_title", "") or c.get("desired_title", "")
    if title:
        parts.extend([title] * 3)

    # Experience descriptions — key for "shipped at scale" signal
    for exp in c.get("experience", []):
        role_title = exp.get("title", "")
        description = exp.get("description", "") or ""
        company = exp.get("company", "") or ""
        if role_title:
            parts.extend([role_title] * 2)
        if description:
            parts.append(description[:300])
        if company:
            parts.append(company)

    # Skills
    for s in c.get("skills", []):
        name = (s.get("name","") if isinstance(s,dict) else s)
        if name:
            parts.append(name)

    # Education
    for edu in c.get("education", []):
        parts.append(edu.get("degree",""))
        parts.append(edu.get("field",""))

    # Summary
    summary = c.get("summary","") or c.get("bio","") or ""
    if summary:
        parts.append(summary[:400])

    return " ".join(filter(None, parts))[:1200]


def extract_skill_names(c: dict) -> set:
    skill_names = set()
    for s in c.get("skills", []):
        name = (s.get("name","") if isinstance(s,dict) else s or "")
        skill_names.add(name.lower().strip())
    return skill_names


def skills_match_score(candidate_skills: set) -> float:
    """
    Weighted skill overlap.
    Required = 1.0 weight, Optional = 0.5 weight.
    Penalty per missing required skill.
    """
    if not JD_REQUIRED_SKILLS:
        return 0.5

    req_matched = len(JD_REQUIRED_SKILLS & candidate_skills)
    opt_matched = len(JD_OPTIONAL_SKILLS & candidate_skills)
    req_missing = len(JD_REQUIRED_SKILLS) - req_matched

    total_weight   = len(JD_REQUIRED_SKILLS)*1.0 + len(JD_OPTIONAL_SKILLS)*0.5
    matched_weight = req_matched*1.0 + opt_matched*0.5

    penalty = (req_missing / max(len(JD_REQUIRED_SKILLS),1)) * 0.15
    score   = matched_weight / total_weight - penalty
    return round(max(0.0, min(1.0, score)), 4)


def experience_score(c: dict, required_years: float = 6.0) -> float:
    """
    JD wants 5-9 years. Sweet spot 6-8 years.
    Too little (<3 yrs) OR too much (>15 yrs, may be consulting) penalised.
    """
    exp_list = c.get("experience", [])
    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    total_years  = total_months / 12.0

    if total_years <= 0:
        total_years = float(c.get("years_of_experience", 0) or 0)

    if total_years < 1:
        return 0.1
    elif total_years < 3:
        return 0.4
    elif total_years <= 9:
        # Sweet spot — linear scale within 3-9 yrs
        return round(min((total_years - 3) / 6.0 * 0.7 + 0.4, 1.0), 4)
    elif total_years <= 12:
        return 0.8
    else:
        return 0.65   # very senior — slight penalty per JD signals


def consulting_penalty(c: dict) -> float:
    """
    JD explicitly: consulting-firms-only = disqualifier.
    Check if ALL experience is at consulting firms.
    """
    exp_list = c.get("experience", [])
    if not exp_list:
        return 0.0

    consulting_count = 0
    for e in exp_list:
        company = (e.get("company","") or "").lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            consulting_count += 1

    ratio = consulting_count / len(exp_list)
    if ratio >= 0.9:    # almost entirely consulting
        return 0.30
    elif ratio >= 0.6:
        return 0.15
    else:
        return 0.0


def product_company_bonus(c: dict) -> float:
    """
    JD wants product company experience — bonus for it.
    """
    exp_list = c.get("experience", [])
    for e in exp_list:
        company = (e.get("company","") or "").lower()
        desc    = (e.get("description","") or "").lower()
        combined = company + " " + desc
        if any(prod in combined for prod in PRODUCT_COMPANIES):
            return 0.08
    return 0.0


def notice_period_score(c: dict) -> float:
    """
    JD: prefers sub-30-day notice. 30+ day = higher bar.
    """
    sig = c.get("redrob_signals", {})
    for field in ("notice_period_days", "notice_period", "notice_days"):
        val = sig.get(field)
        if val is not None:
            days = float(val)
            if days <= 0:
                return 0.10   # immediately available
            elif days <= 30:
                return 0.08
            elif days <= 60:
                return 0.04
            else:
                return 0.0
    return 0.05   # unknown — neutral


def behavioral_score(c: dict) -> float:
    """
    JD explicitly: "perfect-on-paper candidate who hasn't logged in
    for 6 months... is not actually available. Down-weight them."
    Use all available redrob_signals fields adaptively.
    """
    sig = c.get("redrob_signals", {})
    if not sig:
        return 0.25

    scores = []

    # Recency — most important per JD
    for field in ("days_since_last_active","last_active_days","days_inactive"):
        val = sig.get(field)
        if val is not None:
            days = float(val)
            if   days <= 7:   scores.append((1.00, 2.0))  # (score, weight)
            elif days <= 14:  scores.append((0.85, 2.0))
            elif days <= 30:  scores.append((0.65, 2.0))
            elif days <= 90:  scores.append((0.35, 2.0))
            elif days <= 180: scores.append((0.15, 2.0))
            else:             scores.append((0.05, 2.0))  # 6+ months = JD warning
            break

    # Profile completeness
    for field in ("profile_completeness","profile_score","completeness_score"):
        val = sig.get(field)
        if val is not None:
            scores.append((min(float(val)/100.0, 1.0), 1.0))
            break

    # Engagement
    for field in ("platform_engagement_score","engagement_score","engagement"):
        val = sig.get(field)
        if val is not None:
            scores.append((min(float(val)/100.0, 1.0), 1.5))
            break

    # Job seeking intent
    for field in ("job_seeking_intent","open_to_work","actively_looking","is_active"):
        val = sig.get(field)
        if val is not None:
            if isinstance(val, bool):
                scores.append((1.0 if val else 0.2, 1.5))
            elif isinstance(val, (int,float)):
                scores.append((min(float(val)/100.0, 1.0), 1.5))
            break

    # Recruiter response rate
    for field in ("recruiter_response_rate","response_rate"):
        val = sig.get(field)
        if val is not None:
            scores.append((min(float(val)/100.0, 1.0), 1.0))
            break

    # Applications submitted
    for field in ("applications_submitted","applications_count","num_applications"):
        val = sig.get(field)
        if val is not None:
            apps = float(val)
            if   apps >= 10: scores.append((1.0, 0.5))
            elif apps >= 5:  scores.append((0.7, 0.5))
            elif apps >= 2:  scores.append((0.4, 0.5))
            else:            scores.append((0.1, 0.5))
            break

    if not scores:
        return 0.25

    total_weight  = sum(w for _, w in scores)
    weighted_sum  = sum(s*w for s, w in scores)
    return round(weighted_sum / total_weight, 4)


def location_score(c: dict) -> float:
    """JD: Pune/Noida preferred; Delhi NCR, Hyderabad, Mumbai, Bangalore OK."""
    loc = (c.get("location","") or c.get("city","") or "").lower()
    if any(city in loc for city in ("pune","noida")):
        return 0.08
    if any(city in loc for city in ("delhi","ncr","gurgaon","gurugram",
                                     "hyderabad","mumbai","bangalore","bengaluru")):
        return 0.04
    if "india" in loc:
        return 0.02
    return 0.0

# ─────────────────────────────────────────────────────
# STEP 4 — REASONING GENERATOR
# ─────────────────────────────────────────────────────
def generate_reasoning(c: dict, candidate_skills: set,
                        sem_score: float, final_score: float,
                        rank: int) -> str:
    """
    Specific 1-2 sentence reasoning grounded in actual profile facts.
    Submission spec: no hallucination, specific facts, honest concerns,
    rank-consistent tone.
    """
    title      = c.get("current_title","") or c.get("desired_title","")
    exp_list   = c.get("experience", [])
    sig        = c.get("redrob_signals", {})

    # Years of experience
    total_months = sum((e.get("duration_months") or 0) for e in exp_list)
    total_years  = round(total_months / 12.0, 1)
    if total_years == 0:
        total_years = float(c.get("years_of_experience",0) or 0)

    # Matched skills
    matched_req = sorted(JD_REQUIRED_SKILLS & candidate_skills)[:3]
    missing_req = sorted(JD_REQUIRED_SKILLS - candidate_skills)[:2]

    # Location
    location = c.get("location","") or c.get("city","")

    # Recency
    days_inactive = None
    for field in ("days_since_last_active","last_active_days","days_inactive"):
        val = sig.get(field)
        if val is not None:
            days_inactive = float(val)
            break

    # Product company check
    has_product_exp = any(
        any(prod in (e.get("company","") or "").lower()
            for prod in PRODUCT_COMPANIES)
        for e in exp_list
    )

    # Build sentence 1
    parts_s1 = []
    if title and total_years > 0:
        parts_s1.append(f"{total_years}-year {title}")
    elif title:
        parts_s1.append(title)
    elif total_years > 0:
        parts_s1.append(f"{total_years} years experience")

    if matched_req:
        parts_s1.append(f"with strong match on {', '.join(matched_req)}")

    if has_product_exp:
        parts_s1.append("with product company background")

    if location:
        parts_s1.append(f"based in {location}")

    s1 = (", ".join(parts_s1) or "Candidate") + "."

    # Build sentence 2 — concerns and rank tone
    concerns = []
    if missing_req:
        concerns.append(f"missing required: {', '.join(missing_req)}")
    if days_inactive and days_inactive > 180:
        concerns.append(f"inactive for {int(days_inactive)} days (availability concern)")
    elif days_inactive and days_inactive > 90:
        concerns.append(f"low recent activity ({int(days_inactive)} days)")

    # Consulting penalty note
    for e in exp_list:
        company = (e.get("company","") or "").lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            concerns.append("consulting-heavy background")
            break

    if rank <= 10:
        tone = "Strong overall fit per JD requirements"
    elif rank <= 25:
        tone = "Good fit with solid signal alignment"
    elif rank <= 50:
        tone = "Moderate fit"
    elif rank <= 75:
        tone = "Partial fit"
    else:
        tone = "Below threshold — included to complete top-100"

    if concerns:
        s2 = f"{tone}; concerns: {'; '.join(concerns)}."
    else:
        s2 = f"{tone} — no major gaps identified."

    return f"{s1} {s2}"

# ─────────────────────────────────────────────────────
# STEP 5 — MAIN PIPELINE
# ─────────────────────────────────────────────────────
def rank_candidates(candidates_path: str, output_path: str):
    t0 = time.time()

    # 1. Load
    candidates = load_candidates(candidates_path)

    # 2. Honeypot filter
    print("🔍 Filtering honeypots...")
    clean = [(i,c) for i,c in enumerate(candidates) if not is_honeypot(c)]
    print(f"   Removed {len(candidates)-len(clean)} honeypots — {len(clean):,} remain")

    indices, valid_candidates = zip(*clean)

    # 3. Build text representations
    print("📝 Building text representations...")
    texts = [extract_text_for_embedding(c) for c in valid_candidates]

    # 4. Semantic scoring
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Encoding JD...")
    jd_vec = model.encode([JOB_DESCRIPTION], show_progress_bar=False)

    print(f"⚡ Encoding {len(texts):,} candidates...")
    cand_vecs = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    sem_scores = cosine_similarity(jd_vec, cand_vecs)[0]
    print(f"   Encoding done in {time.time()-t0:.1f}s")

    # 5. Feature scores
    print("📊 Computing feature scores...")
    skill_scores, exp_scores, beh_scores = [], [], []
    loc_scores, consult_penalties        = [], []
    prod_bonuses, notice_scores          = [], []
    skill_sets = []

    for c in valid_candidates:
        sk = extract_skill_names(c)
        skill_sets.append(sk)
        skill_scores.append(skills_match_score(sk))
        exp_scores.append(experience_score(c))
        beh_scores.append(behavioral_score(c))
        loc_scores.append(location_score(c))
        consult_penalties.append(consulting_penalty(c))
        prod_bonuses.append(product_company_bonus(c))
        notice_scores.append(notice_period_score(c))

    skill_scores      = np.array(skill_scores)
    exp_scores        = np.array(exp_scores)
    beh_scores        = np.array(beh_scores)
    loc_scores        = np.array(loc_scores)
    consult_penalties = np.array(consult_penalties)
    prod_bonuses      = np.array(prod_bonuses)
    notice_scores     = np.array(notice_scores)

    # 6. Final score
    # Weights: semantic 35%, skills 30%, experience 15%, behavioral 20%
    # Bonuses/penalties on top
    final_scores = (
        sem_scores   * 0.35 +
        skill_scores * 0.30 +
        exp_scores   * 0.15 +
        beh_scores   * 0.20
        + loc_scores
        + prod_bonuses
        + notice_scores
        - consult_penalties
    )
    final_scores = np.clip(final_scores, 0.0, 1.5)

    # 7. Top 100
    top_indices = np.argsort(final_scores)[::-1][:TOP_K]

    # 8. Build output
    print("✍️  Generating reasoning...")
    rows = []
    for rank_pos, idx in enumerate(top_indices, start=1):
        c      = valid_candidates[idx]
        cid    = c.get("candidate_id", c.get("id", f"CAND_{idx:07d}"))
        fscore = round(float(final_scores[idx]), 4)
        sscore = round(float(sem_scores[idx]), 4)
        reason = generate_reasoning(c, skill_sets[idx], sscore, fscore, rank_pos)
        rows.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        fscore,
            "reasoning":    reason
        })

    # 9. Ensure monotonically non-increasing scores
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    # 10. Save
    df = pd.DataFrame(rows, columns=["candidate_id","rank","score","reasoning"])
    df.to_csv(output_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"📄 Saved to: {output_path}")
    print(f"🔢 Score range: {df['score'].max():.4f} → {df['score'].min():.4f}")
    print("\nTop 5 preview:")
    print(df.head().to_string(index=False))

# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India Runs — Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out",        default="./submission.csv")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)