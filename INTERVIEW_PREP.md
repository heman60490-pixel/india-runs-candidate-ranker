# Interview Preparation — India Runs Hackathon 2026

Technical Q&A for judges interview if selected as top finalist.

---

## Architecture Questions

### Q1: Walk me through your complete architecture.
Two-stage pipeline:
1. TF-IDF pre-filter (7.5s): 100K → top 5,000
2. BERT encoding (44s): all-MiniLM-L6-v2 on 5,000
3. Cosine similarity: semantic scores
4. Multi-signal scoring: Semantic(35%) + Skills(30%) + Experience(15%) + Behavioral(20%)
5. Bonuses/Penalties: product +0.08, consulting -0.35, location +0.08
6. Score normalization: 0-1 range
7. Top 100 with specific reasoning

---

### Q2: Why TF-IDF + BERT two-stage?
Direct BERT on 100K = 36 minutes — violates 5-min limit.
TF-IDF filters to 5K in 7.5s, BERT encodes 5K in 44s.
Total: 51 seconds — 34x faster.
A/B test: BERT only NDCG 0.8821 vs TF-IDF+BERT 0.9306.

---

### Q3: Why all-MiniLM-L6-v2 over BGE?
BGE NDCG: 0.9808 but spread: 0.264
MiniLM NDCG: 0.9306 but spread: 0.497
Better candidate differentiation with MiniLM.

---

### Q4: How did you handle JD disqualifiers?
consulting_penalty() function:
- 90%+ consulting exp: -0.35
- 60%+ consulting: -0.20
- 30%+ consulting: -0.08
Firms: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini

---

### Q5: How did you use redrob_signals?
6 signal types with weights:
1. Recency (2.5x) — JD warning: 6+ months inactive = unavailable
2. Profile completeness (1.0x)
3. Platform engagement (1.5x)
4. Job seeking intent (2.0x)
5. Recruiter response rate (1.0x)
6. Applications submitted (0.5x)

---

### Q6: Honeypot detection — 7 rules:
1. >14 expert skills
2. 100% profile + zero experience
3. Single job >25 years
4. Zero engagement + 50+ applications
5. Non-tech title + 3+ AI keywords
6. >30 total skills (keyword stuffer)
7. All skills marked expert

---

### Q7: Runtime improvement story?
v1.0: 36 minutes → v1.1: 51 seconds = 34x faster
Key: TF-IDF O(n log n) pre-filter before BERT O(n)

---

### Q8: Ranking quality validation?
1. validate_submission.py: format check
2. Simulated NDCG@10: 0.6511
3. Labeled 50 candidates manually:
   NDCG@10: 1.0000, MRR: 1.0000, P@5: 1.0000

---

### Q9: What would you improve with more time?
1. BGE-M3 model for better multilingual support
2. Learning-to-rank with recruiter feedback
3. Real NDCG from actual hire/reject decisions
4. Query expansion for JD synonyms
5. Ensemble: MiniLM + BGE scores combined

---

### Q10: Hallucination prevention?
Strict grounding — only real candidate fields used:
- Title from current_title field only
- Years from experience[].duration_months
- Skills from skills[] array only
- Missing skills explicitly stated, never hidden

---

## Links
- GitHub: https://github.com/heman60490-pixel/india-runs-candidate-ranker
- Live Demo: https://india-runs-candidate-ranker.onrender.com/
- Demo Video: https://youtu.be/zacFJfqlO9Q