from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import os

# ✅ FIX 1: Lazy loading — model sirf tab load hoga jab pehli baar use hoga
# Django startup pe download nahi hoga
_model = None

def get_model():
    global _model
    if _model is None:
        print("⏳ Loading SentenceTransformer model (first time only)...")
        _model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("✅ Model loaded successfully!")
    return _model


INDIAN_CERTS = [
    'nptel', 'coursera', 'infosys springboard',
    'tcs ion', 'wipro talentpro', 'nasscom'
]

ROLE_LEVELS = {
    'junior': ['junior', 'entry level', 'fresher', 'trainee', '0-1 year', '1 year'],
    'mid':    ['mid', 'intermediate', '2-3 years', '2 years', '3 years'],
    'senior': ['senior', 'lead', 'principal', 'architect', '5+ years', '4 years', '5 years'],
}

DOMAINS = {
    'backend':  ['django', 'flask', 'fastapi', 'nodejs', 'spring', 'rest api', 'postgresql', 'mongodb'],
    'frontend': ['react', 'javascript', 'html', 'css', 'vue', 'angular'],
    'ml':       ['machine learning', 'deep learning', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
    'devops':   ['docker', 'aws', 'kubernetes', 'ci/cd', 'jenkins'],
}

SKILLS_LIST = [
    'python', 'django', 'javascript', 'react', 'sql',
    'machine learning', 'html', 'css', 'rest api',
    'postgresql', 'mongodb', 'docker', 'aws', 'java',
    'nodejs', 'flask', 'fastapi', 'pandas', 'numpy'
]


def extract_skills(text):
    """Resume ya JD text se skills extract karo."""
    if not text or not isinstance(text, str):
        return set()
    text_lower = text.lower()
    return set(s for s in SKILLS_LIST if s in text_lower)


def skills_score(jd_text, resume_text):
    """JD aur resume ke beech skill match ka score (0.0 - 1.0)."""
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return 0.0
    matched = jd_skills & extract_skills(resume_text)
    return round(len(matched) / len(jd_skills), 4)


def experience_score(resume_text, required_years=2):
    """Resume mein mentioned years of experience ka score (0.0 - 1.0)."""
    if not resume_text or not isinstance(resume_text, str):
        return 0.0
    numbers = re.findall(r'\b(\d+)\s*year', resume_text.lower())
    if numbers:
        found_years = max(int(n) for n in numbers)
        return round(min(found_years / required_years, 1.0), 4)
    return 0.0


def india_bonus(resume_text):
    """Indian certifications ke liye bonus score (max 0.10)."""
    if not resume_text or not isinstance(resume_text, str):
        return 0.0
    text_lower = resume_text.lower()
    bonus = sum(0.05 for cert in INDIAN_CERTS if cert in text_lower)
    return round(min(bonus, 0.10), 4)


def behavioral_score(row):
    """
    Candidate ki activity, profile completeness, aur applications
    ke basis pe behavioral score (0.0 - 1.0).
    """
    # Activity score
    days = row.get('last_active_days', 999)
    if days <= 7:
        activity = 1.0
    elif days <= 30:
        activity = 0.7
    elif days <= 90:
        activity = 0.4
    else:
        activity = 0.1

    # Profile completeness score
    profile = min(row.get('profile_score', 0), 100) / 100

    # Applications count score
    apps = row.get('applications_count', 0)
    if apps >= 10:
        app_score = 1.0
    elif apps >= 5:
        app_score = 0.7
    elif apps >= 2:
        app_score = 0.4
    else:
        app_score = 0.1

    score = (activity * 0.40) + (profile * 0.35) + (app_score * 0.25)
    return round(score, 4)


def detect_level(jd_text):
    """JD se role level detect karo (junior/mid/senior)."""
    if not jd_text:
        return 'mid'
    text_lower = jd_text.lower()
    for level, keywords in ROLE_LEVELS.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return 'mid'


def detect_domain(jd_text):
    """JD se domain detect karo (backend/frontend/ml/devops)."""
    if not jd_text:
        return 'general'
    text_lower = jd_text.lower()
    domain_scores = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in DOMAINS.items()
    }
    best = max(domain_scores, key=domain_scores.get)
    return best if domain_scores[best] > 0 else 'general'


def explain_ranking(jd_text, resume_text, score):
    """Candidate ke ranking ka human-readable explanation."""
    jd_skills     = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    reasons = []
    if matched:
        reasons.append(f"Matched: {', '.join(sorted(matched))}")
    if missing:
        reasons.append(f"Missing: {', '.join(sorted(missing))}")

    if score > 0.7:
        reasons.append("Strong match")
    elif score > 0.4:
        reasons.append("Moderate match")
    else:
        reasons.append("Weak match")

    return " | ".join(reasons)


def parse_jd(jd_text):
    """JD ko parse karke structured dict return karo."""
    return {
        'required_skills':  list(extract_skills(jd_text)),
        'experience_years': round(experience_score(jd_text) * 2, 2),
        'role_level':       detect_level(jd_text),
        'domain':           detect_domain(jd_text),
        'skill_count':      len(extract_skills(jd_text))
    }


def rank_candidates(job_description, candidates_df):
    """
    Candidates ko job description ke against rank karo.

    Args:
        job_description (str): Job ki full description
        candidates_df (pd.DataFrame): Candidates ka DataFrame
            Required columns: resume_text
            Optional columns: last_active_days, profile_score, applications_count

    Returns:
        pd.DataFrame: Ranked candidates with scores and explanations
    """

    # ✅ FIX 2: Model yahan load hoga — sirf jab ranking ho
    model = get_model()

    # ✅ FIX 3: Input validation
    if candidates_df is None or candidates_df.empty:
        print("⚠️  Warning: candidates_df empty hai!")
        return pd.DataFrame()

    if not job_description or not isinstance(job_description, str):
        print("⚠️  Warning: job_description invalid hai!")
        return candidates_df

    # JD Analysis
    jd_analysis = parse_jd(job_description)
    print(f"📋 JD Analysis: {jd_analysis}")

    # ✅ FIX 4: resume_text column check
    if 'resume_text' not in candidates_df.columns:
        raise ValueError("DataFrame mein 'resume_text' column nahi hai!")

    # Embeddings calculate karo
    print(f"🔄 {len(candidates_df)} candidates process ho rahe hain...")
    jd_embedding     = model.encode([job_description])
    resume_texts     = candidates_df['resume_text'].fillna('').tolist()
    resume_embeddings = model.encode(resume_texts, show_progress_bar=False)

    # Semantic similarity
    semantic_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]

    # Baaki scores
    skill_scores      = candidates_df['resume_text'].apply(
        lambda r: skills_score(job_description, str(r) if pd.notna(r) else '')
    )
    exp_scores        = candidates_df['resume_text'].apply(
        lambda r: experience_score(str(r) if pd.notna(r) else '')
    )
    bonus_scores      = candidates_df['resume_text'].apply(
        lambda r: india_bonus(str(r) if pd.notna(r) else '')
    )
    behavioral_scores = candidates_df.apply(behavioral_score, axis=1)

    # ✅ FIX 5: Original df modify nahi hoga
    result_df = candidates_df.copy()
    result_df['semantic_score']   = semantic_scores.round(4)
    result_df['skills_score']     = skill_scores.round(4)
    result_df['experience_score'] = exp_scores.round(4)
    result_df['india_bonus']      = bonus_scores.round(4)
    result_df['behavioral_score'] = behavioral_scores.round(4)

    # Final weighted score
    result_df['final_score'] = (
        semantic_scores        * 0.40 +
        skill_scores           * 0.25 +
        exp_scores             * 0.15 +
        behavioral_scores      * 0.20
    ).round(4)

    # Explanation
    result_df['explanation'] = result_df.apply(
        lambda row: explain_ranking(
            job_description,
            str(row['resume_text']) if pd.notna(row.get('resume_text')) else '',
            row['final_score']
        ),
        axis=1
    )

    # Sort by final score
    ranked = result_df.sort_values('final_score', ascending=False).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)

    print(f"✅ Ranking complete! Top candidate score: {ranked['final_score'].iloc[0]:.4f}")
    return ranked