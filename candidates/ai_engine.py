from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# Model ek baar load karo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Indian certifications
INDIAN_CERTS = [
    'nptel', 'coursera', 'infosys springboard',
    'tcs ion', 'wipro talentpro', 'nasscom'
]

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

def rank_candidates(job_description, candidates_df):
    # Step 1: Embeddings
    jd_embedding = model.encode([job_description])
    resume_texts = candidates_df['resume_text'].fillna('').tolist()
    resume_embeddings = model.encode(resume_texts)

    # Step 2: Scores nikalo
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

    # Step 3: Final score — weighted
    candidates_df = candidates_df.copy()
    candidates_df['semantic_score'] = semantic_scores.round(4)
    candidates_df['skills_score'] = skill_scores.round(4)
    candidates_df['experience_score'] = exp_scores.round(4)
    candidates_df['final_score'] = (
        semantic_scores * 0.50 +
        skill_scores    * 0.30 +
        exp_scores      * 0.20 +
        bonus_scores    * 0.10
    ).round(4)

    # Step 4: Explanation add karo
    candidates_df['explanation'] = candidates_df.apply(
        lambda row: explain_ranking(
            job_description,
            str(row['resume_text']),
            row['final_score']
        ), axis=1
    )

    # Step 5: Sort aur rank
    ranked = candidates_df.sort_values(
        'final_score', ascending=False
    ).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)

    return ranked