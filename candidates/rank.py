"""
rank.py — India Runs Hackathon Submission Script
================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a top-100 ranked CSV with columns:
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

# ─────────────────────────────────────────────
# JOB DESCRIPTION  (paste full JD here)
# ─────────────────────────────────────────────
JOB_DESCRIPTION = """
AI/ML Engineer — Intelligent Search & Retrieval
Redrob AI, Bangalore (Hybrid)

We are looking for an experienced AI/ML Engineer to build intelligent
candidate search and ranking systems. You will design and ship production
NLP/ML pipelines that power Redrob's hiring intelligence platform.

Required Skills:
- Python (3+ years)
- Machine Learning / NLP
- Information Retrieval / Semantic Search
- Vector embeddings (sentence-transformers, FAISS, or similar)
- Experience with large-scale data pipelines

Good to Have:
- RAG systems
- LLM fine-tuning
- FastAPI / Flask
- Docker, AWS

Experience: 3-6 years preferred
Location: Bangalore preferred; open to remote for strong candidates
"""

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TOP_K = 100          # submission requires exactly 100
MAX_RUNTIME_SECS = 270   # stay well under 5-min wall-clock limit

# Skills the JD cares about (extend after reading job_description.md)
JD_REQUIRED_SKILLS = {
    "python", "machine learning", "nlp", "information retrieval",
    "semantic search", "vector embeddings", "sentence-transformers",
    "faiss", "data pipelines", "pytorch", "tensorflow", "scikit-learn"
}
JD_OPTIONAL_SKILLS = {
    "rag", "llm", "fastapi", "flask", "docker", "aws",
    "kubernetes", "transformers", "huggingface"
}

# Honeypot detection thresholds
MAX_SKILLS_EXPERT = 12      # >12 "expert" skills → suspicious
MIN_COMPANY_AGE_YEARS = 1   # company founded after candidate's start → impossible

# ─────────────────────────────────────────────
# STEP 1 — LOAD CANDIDATES
# ─────────────────────────────────────────────
def load_candidates(path: str) -> list[dict]:
    """Load candidates.jsonl (plain or gzipped)."""
    print(f"📂 Loading candidates from: {path}")
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

# ─────────────────────────────────────────────
# STEP 2 — HONEYPOT DETECTION
# ─────────────────────────────────────────────
def is_honeypot(c: dict) -> bool:
    """
    Detect impossible/fake profiles.
    Submission spec: >10% honeypots in top-100 = disqualification.
    """
    signals = c.get("redrob_signals", {})
    skills  = c.get("skills", [])

    # Too many expert-level skills
    expert_skills = [s for s in skills if isinstance(s, dict) and
                     s.get("proficiency", "").lower() in ("expert", "advanced")]
    if len(expert_skills) > MAX_SKILLS_EXPERT:
        return True

    # Profile completeness 100 but no experience
    experience = c.get("experience", [])
    if signals.get("profile_completeness", 0) == 100 and len(experience) == 0:
        return True

    # Claimed years at company exceed company age
    for exp in experience:
        duration = exp.get("duration_months", 0)
        if duration and duration > 240:   # > 20 years at one company
            return True

    # Engagement = 0 but applied to 50+ jobs (bot signal)
    engagement = signals.get("platform_engagement_score", 0)
    apps = signals.get("applications_submitted", 0)
    if engagement == 0 and apps > 50:
        return True

    return False

# ─────────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_text_for_embedding(c: dict) -> str:
    """
    Build a rich text blob from candidate fields for semantic embedding.
    Weights important fields by repetition.
    """
    parts = []

    # Current title (repeated for weight)
    title = c.get("current_title", "") or c.get("desired_title", "")
    if title:
        parts.extend([title] * 3)

    # Skills
    for s in c.get("skills", []):
        if isinstance(s, dict):
            parts.append(s.get("name", ""))
        elif isinstance(s, str):
            parts.append(s)

    # Experience summaries
    for exp in c.get("experience", []):
        parts.append(exp.get("title", ""))
        parts.append(exp.get("description", ""))

    # Education
    for edu in c.get("education", []):
        parts.append(edu.get("degree", ""))
        parts.append(edu.get("field", ""))

    # Summary / bio
    parts.append(c.get("summary", "") or c.get("bio", ""))

    return " ".join(filter(None, parts))[:1000]   # cap at 1000 chars


def extract_skill_names(c: dict) -> set:
    """Return lowercase skill names from candidate."""
    skill_names = set()
    for s in c.get("skills", []):
        if isinstance(s, dict):
            name = s.get("name", "").lower()
        elif isinstance(s, str):
            name = s.lower()
        else:
            continue
        skill_names.add(name)
    return skill_names


def skills_match_score(candidate_skills: set) -> float:
    """
    Weighted skill overlap against JD.
    Required skills worth 1.0, optional 0.5.
    Penalise missing required skills.
    """
    if not JD_REQUIRED_SKILLS and not JD_OPTIONAL_SKILLS:
        return 0.5

    req_matched  = len(JD_REQUIRED_SKILLS  & candidate_skills)
    opt_matched  = len(JD_OPTIONAL_SKILLS  & candidate_skills)
    req_missing  = len(JD_REQUIRED_SKILLS) - req_matched

    total_weight = len(JD_REQUIRED_SKILLS) * 1.0 + len(JD_OPTIONAL_SKILLS) * 0.5
    if total_weight == 0:
        return 0.5

    matched_weight = req_matched * 1.0 + opt_matched * 0.5

    # Penalty per missing required skill
    penalty = (req_missing / max(len(JD_REQUIRED_SKILLS), 1)) * 0.15
    score = matched_weight / total_weight - penalty
    return round(max(0.0, min(1.0, score)), 4)


def experience_score(c: dict, required_years: float = 3.0) -> float:
    """Score based on total years of experience."""
    exp_list = c.get("experience", [])
    total_months = sum(e.get("duration_months", 0) or 0 for e in exp_list)
    total_years  = total_months / 12.0

    if total_years <= 0:
        # Fallback: try parsing years_of_experience field
        yoe = c.get("years_of_experience", 0) or 0
        total_years = float(yoe)

    score = min(total_years / required_years, 1.0)
    return round(score, 4)


def behavioral_score(c: dict) -> float:
    """
    Use redrob_signals for behavioral scoring.
    Adapts to whatever field names exist in the real dataset.
    """
    sig = c.get("redrob_signals", {})
    if not sig:
        return 0.3   # no signals → neutral

    scores = []

    # Recency / activity
    for field in ("days_since_last_active", "last_active_days", "days_inactive"):
        val = sig.get(field)
        if val is not None:
            if   val <= 7:   scores.append(1.0)
            elif val <= 30:  scores.append(0.7)
            elif val <= 90:  scores.append(0.4)
            else:            scores.append(0.1)
            break

    # Profile completeness
    for field in ("profile_completeness", "profile_score", "completeness_score"):
        val = sig.get(field)
        if val is not None:
            scores.append(min(float(val) / 100.0, 1.0))
            break

    # Platform engagement
    for field in ("platform_engagement_score", "engagement_score", "engagement"):
        val = sig.get(field)
        if val is not None:
            scores.append(min(float(val) / 100.0, 1.0))
            break

    # Applications submitted
    for field in ("applications_submitted", "applications_count", "num_applications"):
        val = sig.get(field)
        if val is not None:
            if   val >= 10: scores.append(1.0)
            elif val >= 5:  scores.append(0.7)
            elif val >= 2:  scores.append(0.4)
            else:           scores.append(0.1)
            break

    # Job seeking intent
    for field in ("job_seeking_intent", "open_to_work", "actively_looking"):
        val = sig.get(field)
        if val is not None:
            if isinstance(val, bool):
                scores.append(1.0 if val else 0.3)
            elif isinstance(val, (int, float)):
                scores.append(min(float(val) / 100.0, 1.0))
            break

    return round(float(np.mean(scores)) if scores else 0.3, 4)


def location_score(c: dict, preferred_city: str = "bangalore") -> float:
    """Bonus for preferred location."""
    location = (c.get("location", "") or c.get("city", "") or "").lower()
    if preferred_city in location:
        return 0.1
    if "india" in location:
        return 0.05
    return 0.0


# ─────────────────────────────────────────────
# STEP 4 — REASONING GENERATOR
# ─────────────────────────────────────────────
def generate_reasoning(c: dict, candidate_skills: set,
                        sem_score: float, final_score: float,
                        rank: int) -> str:
    """
    Generate specific 1-2 sentence reasoning.
    References actual candidate facts — no hallucination.
    """
    name = c.get("full_name", c.get("name", "Candidate"))
    title = c.get("current_title", c.get("desired_title", ""))

    # Experience
    exp_list = c.get("experience", [])
    total_months = sum(e.get("duration_months", 0) or 0 for e in exp_list)
    total_years = round(total_months / 12.0, 1)
    if total_years == 0:
        total_years = c.get("years_of_experience", 0) or 0

    # Matched skills
    matched_req = JD_REQUIRED_SKILLS & candidate_skills
    matched_opt = JD_OPTIONAL_SKILLS & candidate_skills
    missing_req = JD_REQUIRED_SKILLS - candidate_skills

    # Location
    location = c.get("location", c.get("city", ""))

    # Build sentence 1 — who they are
    if title and total_years > 0:
        s1 = f"{total_years}-year {title}"
    elif title:
        s1 = f"{title}"
    elif total_years > 0:
        s1 = f"{total_years} years of experience"
    else:
        s1 = "Candidate"

    if matched_req:
        top_skills = sorted(matched_req)[:3]
        s1 += f" with strong match on {', '.join(top_skills)}"

    if location:
        s1 += f"; based in {location}"

    s1 += "."

    # Build sentence 2 — why this rank
    concerns = []
    if missing_req:
        top_missing = sorted(missing_req)[:2]
        concerns.append(f"missing {', '.join(top_missing)}")

    sig = c.get("redrob_signals", {})
    for field in ("days_since_last_active", "last_active_days"):
        days = sig.get(field)
        if days and days > 90:
            concerns.append("low recent activity")
            break

    if rank <= 10:
        tone = "Strong overall fit for the role"
    elif rank <= 30:
        tone = "Good fit with solid signal alignment"
    elif rank <= 60:
        tone = "Moderate fit"
    else:
        tone = "Partial fit"

    if concerns:
        s2 = f"{tone}; concerns: {'; '.join(concerns)}."
    else:
        s2 = f"{tone} — no major gaps identified."

    return f"{s1} {s2}"


# ─────────────────────────────────────────────
# STEP 5 — MAIN RANKING PIPELINE
# ─────────────────────────────────────────────
def rank_candidates(candidates_path: str, output_path: str):
    t0 = time.time()

    # 1. Load
    candidates = load_candidates(candidates_path)

    # 2. Filter honeypots
    print("🔍 Detecting honeypot candidates...")
    clean = [(i, c) for i, c in enumerate(candidates) if not is_honeypot(c)]
    honeypot_count = len(candidates) - len(clean)
    print(f"   Removed {honeypot_count} honeypots — {len(clean):,} candidates remain")

    # 3. Build text for embedding
    print("📝 Building text representations...")
    indices, valid_candidates = zip(*clean)
    texts = [extract_text_for_embedding(c) for c in valid_candidates]

    # 4. Semantic scoring — fast with MiniLM
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Encoding JD...")
    jd_vec = model.encode([JOB_DESCRIPTION], show_progress_bar=False)

    print(f"⚡ Encoding {len(texts):,} candidates (batch mode)...")
    # Batch encode — crucial for speed
    cand_vecs = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    sem_scores = cosine_similarity(jd_vec, cand_vecs)[0]
    print(f"   Encoding done in {time.time()-t0:.1f}s")

    # 5. Other feature scores
    print("📊 Computing feature scores...")
    skill_scores, exp_scores, beh_scores, loc_scores = [], [], [], []
    skill_sets = []

    for c in valid_candidates:
        sk = extract_skill_names(c)
        skill_sets.append(sk)
        skill_scores.append(skills_match_score(sk))
        exp_scores.append(experience_score(c))
        beh_scores.append(behavioral_score(c))
        loc_scores.append(location_score(c))

    skill_scores = np.array(skill_scores)
    exp_scores   = np.array(exp_scores)
    beh_scores   = np.array(beh_scores)
    loc_scores   = np.array(loc_scores)

    # 6. Final score formula
    final_scores = (
        sem_scores   * 0.35 +
        skill_scores * 0.30 +
        exp_scores   * 0.15 +
        beh_scores   * 0.15 +
        loc_scores   * 0.05
    )

    # 7. Get top 100
    top_indices = np.argsort(final_scores)[::-1][:TOP_K]

    # 8. Build output rows
    print("✍️  Generating reasoning for top 100...")
    rows = []
    for rank_pos, idx in enumerate(top_indices, start=1):
        c    = valid_candidates[idx]
        cid  = c.get("candidate_id", c.get("id", f"CAND_{idx:07d}"))
        fscore = round(float(final_scores[idx]), 4)
        sscore = round(float(sem_scores[idx]), 4)
        reasoning = generate_reasoning(
            c, skill_sets[idx], sscore, fscore, rank_pos
        )
        rows.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        fscore,
            "reasoning":    reasoning
        })

    # 9. Validate monotonically non-increasing scores
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]   # clamp

    # 10. Save CSV
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv(output_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n✅ Done! {TOP_K} candidates ranked in {elapsed:.1f}s")
    print(f"📄 Output saved to: {output_path}")
    print(f"\n🔢 Score range: {df['score'].max():.4f} → {df['score'].min():.4f}")
    print(f"\nTop 5 preview:")
    print(df.head().to_string(index=False))


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="India Runs Hackathon — Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)