from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Model ek baar load karo — har request pe dobara load nahi hoga
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_skills(text):
    """
    Text mein se known skills dhundho
    Example: "I know Python and Django" → {'python', 'django'}
    """
    skills_list = [
        'python', 'django', 'javascript', 'react', 'sql',
        'machine learning', 'html', 'css', 'rest api',
        'postgresql', 'mongodb', 'docker', 'aws', 'java',
        'nodejs', 'flask', 'fastapi', 'pandas', 'numpy'
    ]
    text_lower = text.lower()
    found_skills = set()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.add(skill)
    return found_skills

def skills_score(jd_text, resume_text):
    """
    JD ki skills aur resume ki skills compare karo
    Example: JD mein 3 skills, resume mein 2 match → score = 0.66
    """
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    if not jd_skills:
        return 0.0

    matched = jd_skills.intersection(resume_skills)
    return len(matched) / len(jd_skills)

def rank_candidates(job_description, candidates_df):
    """
    Main function — JD aur candidates ka dataframe lo
    Ranked dataframe wapas do
    """

    # Step 1: JD ko numbers mein convert karo
    jd_embedding = model.encode([job_description])

    # Step 2: Har candidate ke resume ko numbers mein convert karo
    resume_texts = candidates_df['resume_text'].fillna('').tolist()
    resume_embeddings = model.encode(resume_texts)

    # Step 3: Similarity nikalo — 0 to 1 ke beech
    # 1 = perfect match, 0 = bilkul alag
    semantic_scores = cosine_similarity(
        jd_embedding, resume_embeddings
    )[0]

    # Step 4: Skills match score nikalo
    skill_scores = candidates_df['resume_text'].apply(
        lambda resume: skills_score(job_description, str(resume))
    )

    # Step 5: Final score — dono ko milao
    # Semantic = 60% weight, Skills = 40% weight
    candidates_df = candidates_df.copy()
    candidates_df['semantic_score'] = semantic_scores.round(4)
    candidates_df['skills_score'] = skill_scores.round(4)
    candidates_df['final_score'] = (
        semantic_scores * 0.60 +
        skill_scores    * 0.40
    ).round(4)

    # Step 6: Sort karo — highest score pehle
    ranked = candidates_df.sort_values(
        'final_score', ascending=False
    ).reset_index(drop=True)

    # Step 7: Rank number add karo — 1, 2, 3...
    ranked['rank'] = range(1, len(ranked) + 1)

    return ranked