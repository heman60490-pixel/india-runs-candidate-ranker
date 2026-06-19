from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# Model load karo
print("⏳ Loading SentenceTransformer model (first time only)...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("✅ Model loaded successfully!")

# ================================
# CONSTANTS
# ================================

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

SKILLS_LIST = [
    'python', 'django', 'javascript', 'react', 'sql',
    'machine learning', 'html', 'css', 'rest api',
    'postgresql', 'mongodb', 'docker', 'aws', 'java',
    'nodejs', 'flask', 'fastapi', 'pandas', 'numpy',
    'kubernetes', 'tensorflow', 'pytorch', 'vue', 'angular',
    'spring', 'mysql', 'redis', 'celery', 'git'
]

# Education levels
EDUCATION_LEVELS = {
    'phd': ['phd', 'doctorate', 'ph.d'],
    'masters': ['m.tech', 'mtech', 'mca', 'm.sc', 'msc', 'masters', 'post graduate', 'pg'],
    'bachelors': ['b.tech', 'btech', 'bca', 'b.sc', 'bsc', 'be ', 'b.e', 'bachelors', 'graduate'],
    'diploma': ['diploma', 'polytechnic'],
}

# ================================
# SKILL FUNCTIONS
# ================================

def extract_skills(text):
    """Text mein se skills nikalo"""
    text_lower = text.lower()
    return set(s for s in SKILLS_LIST if s in text_lower)

def extract_skills_weighted(jd_text):
    """
    JD mein se required aur optional skills alag detect karo
    Required skills = 1.0 weight
    Nice to have skills = 0.5 weight
    """
    text_lower = jd_text.lower()

    required_patterns = [
        'required', 'must have', 'must-have',
        'mandatory', 'essential', 'need', 'needed'
    ]
    optional_patterns = [
        'good to have', 'nice to have', 'preferred',
        'bonus', 'optional', 'plus', 'advantage',
        'beneficial', 'desirable'
    ]

    all_skills = list(extract_skills(jd_text))
    weighted_skills = {}

    for skill in all_skills:
        skill_pos = text_lower.find(skill)
        if skill_pos == -1:
            weighted_skills[skill] = 1.0
            continue

        context = text_lower[max(0, skill_pos-100):skill_pos]
        is_optional = any(pat in context for pat in optional_patterns)
        is_required = any(pat in context for pat in required_patterns)

        if is_optional:
            weighted_skills[skill] = 0.5
        elif is_required:
            weighted_skills[skill] = 1.0
        else:
            weighted_skills[skill] = 1.0

    return weighted_skills

def skills_score(jd_text, resume_text):
    """Basic skills score"""
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return 0.0
    matched = jd_skills & extract_skills(resume_text)
    return len(matched) / len(jd_skills)

def skills_score_weighted(jd_text, resume_text):
    """
    Weighted skills score
    Required skills zyada important hain
    """
    weighted_jd_skills = extract_skills_weighted(jd_text)
    if not weighted_jd_skills:
        return 0.0

    resume_skills = extract_skills(resume_text)
    total_weight = sum(weighted_jd_skills.values())
    matched_weight = 0.0

    for skill, weight in weighted_jd_skills.items():
        if skill in resume_skills:
            matched_weight += weight

    return round(matched_weight / total_weight, 4) if total_weight > 0 else 0.0

# ================================
# EXPERIENCE FUNCTIONS
# ================================

def experience_score(resume_text, required_years=2):
    """Resume mein se experience years nikalo"""
    numbers = re.findall(r'\b(\d+)\s*year', resume_text.lower())
    if numbers:
        found_years = max(int(n) for n in numbers)
        return min(found_years / required_years, 1.0)
    return 0.0

# ================================
# CAREER METADATA FUNCTIONS
# ================================

def detect_education(resume_text):
    """
    Resume mein se education level detect karo
    PhD > Masters > Bachelors > Diploma
    """
    text_lower = resume_text.lower()
    scores = {
        'phd': 1.0,
        'masters': 0.85,
        'bachelors': 0.70,
        'diploma': 0.50,
        'none': 0.40
    }

    for level, keywords in EDUCATION_LEVELS.items():
        for kw in keywords:
            if kw in text_lower:
                return level, scores[level]

    return 'none', scores['none']

def detect_career_gaps(resume_text):
    """
    Resume mein career gaps detect karo
    Gap = score kam hoga
    """
    text_lower = resume_text.lower()

    # Gap keywords
    gap_keywords = ['gap', 'break', 'career break', 'sabbatical', 'unemployed']
    has_gap = any(kw in text_lower for kw in gap_keywords)

    if has_gap:
        return 0.6  # gap hai toh thoda kam score
    return 1.0      # no gap = full score

def detect_job_switches(resume_text):
    """
    Kitni companies mein kaam kiya — detect karo
    Bahut zyada switches = unstable candidate
    """
    text_lower = resume_text.lower()

    # Company keywords
    company_keywords = ['pvt ltd', 'private limited', 'technologies',
                       'solutions', 'systems', 'infotech', 'software',
                       'ltd', 'inc', 'corp']

    count = sum(1 for kw in company_keywords if kw in text_lower)

    # 1-2 companies = stable = 1.0
    # 3-4 companies = normal = 0.8
    # 5+ companies = unstable = 0.6
    if count <= 2:
        return 1.0
    elif count <= 4:
        return 0.8
    else:
        return 0.6

def career_metadata_score(resume_text):
    """
    3 career signals milake score banao:
    1. Education level
    2. Career gaps
    3. Job switches
    """
    _, edu_score = detect_education(resume_text)
    gap_score = detect_career_gaps(resume_text)
    switch_score = detect_job_switches(resume_text)

    final = (
        edu_score    * 0.50 +
        gap_score    * 0.30 +
        switch_score * 0.20
    )
    return round(final, 4)

# ================================
# BEHAVIORAL FUNCTIONS
# ================================

def behavioral_score(row):
    """
    3 behavioral signals:
    1. Last active days
    2. Profile completeness
    3. Applications count
    """
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

# ================================
# INDIA BONUS
# ================================

def india_bonus(resume_text):
    """Indian certifications ka bonus"""
    text_lower = resume_text.lower()
    bonus = 0.0
    for cert in INDIAN_CERTS:
        if cert in text_lower:
            bonus += 0.05
    return min(bonus, 0.10)

# ================================
# JD PARSER
# ================================

def detect_level(jd_text):
    """JD mein se role level detect karo"""
    text_lower = jd_text.lower()
    for level, keywords in ROLE_LEVELS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return 'mid'

def detect_domain(jd_text):
    """JD mein se domain detect karo"""
    text_lower = jd_text.lower()
    domain_scores = {}
    for domain, keywords in DOMAINS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        domain_scores[domain] = score
    best = max(domain_scores, key=domain_scores.get)
    return best if domain_scores[best] > 0 else 'general'

def parse_jd(jd_text):
    """JD ko deeply parse karo"""
    weighted_skills = extract_skills_weighted(jd_text)
    required = [s for s, w in weighted_skills.items() if w == 1.0]
    optional = [s for s, w in weighted_skills.items() if w == 0.5]

    return {
        'required_skills': required,
        'optional_skills': optional,
        'experience_years': experience_score(jd_text) * 2,
        'role_level': detect_level(jd_text),
        'domain': detect_domain(jd_text),
        'skill_count': len(weighted_skills)
    }

# ================================
# EXPLAINABILITY
# ================================

def explain_ranking(jd_text, resume_text, score):
    """Kyun rank hua — explain karo"""
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    edu_level, _ = detect_education(resume_text)

    reasons = []
    if matched:
        reasons.append(f"Matched: {', '.join(matched)}")
    if missing:
        reasons.append(f"Missing: {', '.join(missing)}")
    if edu_level != 'none':
        reasons.append(f"Education: {edu_level}")
    if score > 0.7:
        reasons.append("Strong match")
    elif score > 0.4:
        reasons.append("Moderate match")
    else:
        reasons.append("Weak match")

    return " | ".join(reasons)

# ================================
# MAIN RANKING FUNCTION
# ================================

def rank_candidates(job_description, candidates_df):
    """
    Main function — JD aur candidates ka dataframe lo
    Ranked dataframe wapas do
    """
    # JD parse karo
    jd_analysis = parse_jd(job_description)
    print(f"📋 JD Analysis: {jd_analysis}")
    print(f"🔄 {len(candidates_df)} candidates process ho rahe hain...")

    # Step 1: Embeddings
    jd_embedding = model.encode([job_description])
    resume_texts = candidates_df['resume_text'].fillna('').tolist()
    resume_embeddings = model.encode(resume_texts)

    # Step 2: Semantic similarity
    semantic_scores = cosine_similarity(
        jd_embedding, resume_embeddings
    )[0]

    # Step 3: Weighted skills score
    skill_scores = candidates_df['resume_text'].apply(
        lambda r: skills_score_weighted(job_description, str(r))
    )

    # Step 4: Experience score
    exp_scores = candidates_df['resume_text'].apply(
        lambda r: experience_score(str(r))
    )

    # Step 5: Behavioral score
    behavioral_scores = candidates_df.apply(
        lambda row: behavioral_score(row), axis=1
    )

    # Step 6: Career metadata score
    career_scores = candidates_df['resume_text'].apply(
        lambda r: career_metadata_score(str(r))
    )

    # Step 7: India bonus
    bonus_scores = candidates_df['resume_text'].apply(
        lambda r: india_bonus(str(r))
    )

    # Step 8: Final weighted score
    candidates_df = candidates_df.copy()
    candidates_df['semantic_score']  = semantic_scores.round(4)
    candidates_df['skills_score']    = skill_scores.round(4)
    candidates_df['experience_score'] = exp_scores.round(4)
    candidates_df['behavioral_score'] = behavioral_scores.round(4)
    candidates_df['career_score']    = career_scores.round(4)

    candidates_df['final_score'] = (
        semantic_scores   * 0.35 +  # 35% — AI similarity
        skill_scores      * 0.25 +  # 25% — Weighted skills
        exp_scores        * 0.15 +  # 15% — Experience
        behavioral_scores * 0.15 +  # 15% — Behavioral
        career_scores     * 0.10    # 10% — Career metadata
    ).round(4)

    # India bonus add karo
    candidates_df['final_score'] = (
        candidates_df['final_score'] + bonus_scores
    ).round(4)

    # Step 9: Explanation
    candidates_df['explanation'] = candidates_df.apply(
        lambda row: explain_ranking(
            job_description,
            str(row['resume_text']),
            row['final_score']
        ), axis=1
    )

    # Step 10: Sort aur rank
    ranked = candidates_df.sort_values(
        'final_score', ascending=False
    ).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)

    print(f"✅ Ranking complete! Top candidate score: {ranked.iloc[0]['final_score']}")
    return ranked