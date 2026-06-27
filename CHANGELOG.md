# Changelog — India Runs Candidate Ranker

All notable changes to this project are documented here.

---

## v1.4 (27-Jun-2026)
### Added
- Expanded labeled evaluation dataset to 50 candidates
- Added NDCG@5, NDCG@10, NDCG@20, NDCG@50 metrics
- Added Precision@K, Recall@K, F1@K evaluation
- Added A/B testing results (BERT only vs TF-IDF+BERT)
- Added model comparison: MiniLM vs BGE
- Added labeling_methodology.md for ground truth documentation
- Added INTERVIEW_PREP.md for technical interview preparation
- Updated README with evaluation results and model comparison

### Changed
- evaluate.py: expanded from 15 to 50 labeled candidates
- README: updated project structure, sample results, performance

---

## v1.3 (25-Jun-2026)
### Added
- evaluate.py: real NDCG/MRR evaluation with labeled dataset
- model_comparison.py: MiniLM vs BGE comparison
- model_comparison_results.csv: comparison results saved
- submission_metadata.yaml: hackathon metadata

### Changed
- rank.py: score normalization 0-1 for better spread
- rank.py: improved honeypot detection (7 rules)
- rank.py: better reasoning generation (6 rank tones)
- rank.py: added NDCG@10 simulated metric

### Results
- Score spread: 0.05 → 1.0000
- Std deviation: 0.009 → 0.1738
- NDCG@10: 0.6511 (simulated)

---

## v1.2 (24-Jun-2026)
### Added
- TF-IDF pre-filtering stage (100K → 5K in 7.5s)
- Consulting firm penalty (from JD disqualifiers)
- Product company bonus
- Notice period scoring
- Location scoring (Pune/Noida preferred)

### Changed
- rank.py: two-stage pipeline (TF-IDF + BERT)
- JD_REQUIRED_SKILLS: removed non-existent skills
- Experience scoring: 5 brackets matching JD

### Performance
- Runtime: 36 minutes → 51 seconds (34x improvement)

---

## v1.1 (23-Jun-2026)
### Added
- rank.py: standalone submission script
- Multi-signal scoring: Semantic + Skills + Experience + Behavioral
- Honeypot detection (initial 4 rules)
- JD-specific scoring weights

### Changed
- README: added single-command reproduction
- requirements.txt: added openpyxl

---

## v1.0 (20-Jun-2026)
### Initial Release
- Django web application for candidate ranking
- Basic BERT embeddings (all-MiniLM-L6-v2)
- Simple skill matching
- CSV upload and download
- Deployed on Render.com