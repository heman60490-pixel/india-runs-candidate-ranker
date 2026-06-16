from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

INDIAN_CERTS = [
    'nptel', 'coursera', 'infosys springboard',
    'tcs ion', 'wipro talentpro', 'nasscom'
]

ROLE_LEVELS = {
    'junior': ['junior', 'entry level', 'fresher', 'trainee', '0-1 year', '1 year'],
    'mid': ['mid', 'intermediate', '2-3 years', '2 years', '3 years'],
    'senior': ['senior', 'lead', 'principal', 'architect', '5+ years', '4 years', '5 years'],
}

DOMAINS = {
    'backend': ['django', 'flask', 'fastapi', 'nodejs', 'spring', 'rest api', 'postgresql', 'mongodb'],
    'frontend': ['react', 'javascript', 'html', 'css', 'vue', 'angular'],
    'ml': ['machine learning', 'deep learning', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
    'devops': ['docker', 'aws', 'kubernetes', 'ci/cd', 'jenkins'],
}

def extract_skills(text):
    skills_list = [
        'python', 'django', 'javascript', 'react', 'sql',
        'machine learning', 'html', 'css', 'rest api',
        'postgresql', 'mongodb', 'docker', 'aws', 'java',
        'nodejs', 'flask', 'fastapi', 'pandas', 'numpy'
    ]
    text_lower = text.lower()
    return set(s for s in skills_list if s in text_lower)

def skills_score(jd_text, resume_text):
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return 0.0
    matched = jd_skills & extract_skills(resume_text)
    return len(matched) / len(jd_skills)

def experience_score(resume_text, required_years=2):
    numbers = re.findall(r'\b(\d+)\s*year', resume_text.lower())
    if numbers:
        found_years = max(int(n) for n in numbers)
        return min(found_years / required_years, 1.0)
    return 0.0

def india_bonus(resume_text):
    text_lower = resume_text.lower()
    bonus = 0.0
    for cert in INDIAN_CERTS:
        if cert in text_lower:
            bonus += 0.05
    return min(bonus, 0.10)

def behavioral_score(row):
    days = row.get('last_active_days', 999)
    if days <= 7:
        activity = 1.0
    elif days <= 30:
        activity = 0.7
    elif days <= 90:
        activity = 0.4
    else:
        activity = 0.1

    profile = row.get('profile_score', 0) / 100

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
    text_lower = jd_text.lower()
    for level, keywords in ROLE_LEVELS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return 'mid'

def detect_domain(jd_text):
    text_lower = jd_text.lower()
    domain_scores = {}
    for domain, keywords in DOMAINS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        domain_scores[domain] = score
    best = max(domain_scores, key=domain_scores.get)
    return best if domain_scores[best] > 0 else 'general'

def explain_ranking(jd_text, resume_text, score):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    reasons = []
    if matched:
        reasons.append(f"Matched: {', '.join(matched)}")
    if missing:
        reasons.append(f"Missing: {', '.join(missing)}")
    if score > 0.7:
        reasons.append("Strong match")
    elif score > 0.4:
        reasons.append("Moderate match")
    else:
        reasons.append("Weak match")

    return " | ".join(reasons)

def parse_jd(jd_text):
    return {
        'required_skills': list(extract_skills(jd_text)),
        'experience_years': experience_score(jd_text) * 2,
        'role_level': detect_level(jd_text),
        'domain': detect_domain(jd_text),
        'skill_count': len(extract_skills(jd_text))
    }

def rank_candidates(job_description, candidates_df):

    jd_analysis = parse_jd(job_description)
    print(f"JD Analysis: {jd_analysis}")

    jd_embedding = model.encode([job_description])
    resume_texts = candidates_df['resume_text'].fillna('').tolist()
    resume_embeddings = model.encode(resume_texts)

    semantic_scores = cosine_similarity(
        jd_embedding, resume_embeddings
    )[0]

    skill_scores = candidates_df['resume_text'].apply(
        lambda r: skills_score(job_description, str(r))
    )

    exp_scores = candidates_df['resume_text'].apply(
        lambda r: experience_score(str(r))
    )

    bonus_scores = candidates_df['resume_text'].apply(
        lambda r: india_bonus(str(r))
    )

    behavioral_scores = candidates_df.apply(
        lambda row: behavioral_score(row), axis=1
    )

    candidates_df = candidates_df.copy()
    candidates_df['semantic_score'] = semantic_scores.round(4)
    candidates_df['skills_score'] = skill_scores.round(4)
    candidates_df['experience_score'] = exp_scores.round(4)
    candidates_df['behavioral_score'] = behavioral_scores.round(4)

    candidates_df['final_score'] = (
        semantic_scores   * 0.40 +
        skill_scores      * 0.25 +
        exp_scores        * 0.15 +
        behavioral_scores * 0.20
    ).round(4)

    candidates_df['explanation'] = candidates_df.apply(
        lambda row: explain_ranking(
            job_description,
            str(row['resume_text']),
            row['final_score']
        ), axis=1
    )

    ranked = candidates_df.sort_values(
        'final_score', ascending=False
    ).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)

    return ranked