# 🇮🇳 India Runs — AI Candidate Ranker

An AI-powered candidate ranking system built for the **India Runs Hackathon 2026** by Redrob AI.

---

## 🎯 Problem
Recruiters receive thousands of resumes — keyword filters don't work.
Context, skills, and experience need to be understood together.

## 💡 Solution
An intelligent ranking system that combines:
- **Semantic understanding** — BERT embeddings match meaning, not just keywords
- **Skills matching** — exact skill overlap between JD and resume
- **Experience scoring** — years of experience vs requirement
- **India-specific bonus** — recognizes Indian certifications (NPTEL, TCS iON, etc.)
- **Explainability** — tells WHY each candidate was ranked

---

## 🏗️ How It Works
Job Description

↓

BERT Embeddings (all-MiniLM-L6-v2)

↓

┌─────────────────────────────────┐

│ Semantic Score      → 50% weight│

│ Skills Match Score  → 30% weight│

│ Experience Score    → 20% weight│

│ India Bonus         → +10% max  │

└─────────────────────────────────┘

↓

Final Ranked List with Explanation

## 🛠️ Tech Stack
- Python 3.10+
- Django 4.2
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn
- Pandas, NumPy

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/heman60490-pixel/india-runs-candidate-ranker.git
cd india-runs-candidate-ranker

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
python manage.py runserver
```

Open `http://127.0.0.1:8000` in browser.

---

## 📊 Features

| Feature | Description |
|---|---|
| Semantic Ranking | BERT-based meaning match |
| Skills Matching | Exact skill overlap detection |
| Experience Score | Years of experience scoring |
| India Bonus | Indian certification recognition |
| Explainability | Why each candidate was ranked |
| CSV Upload | Upload your own candidate dataset |
| Download Results | Export ranked candidates as CSV |
| Web Interface | Clean Django-powered UI |

---

## 📁 Project Structure
india_runs/

├── candidates/

│   ├── ai_engine.py     ← Core AI ranking logic

│   ├── views.py         ← Request handling

│   ├── urls.py          ← URL routing

│   └── templates/

│       ├── index.html   ← JD input form

│       └── results.html ← Ranked results

├── data/

│   └── candidates.csv   ← Dataset

├── manage.py

├── requirements.txt

└── README.md

---

## 📈 Sample Results

| Rank | Candidate | Final Score | Explanation |
|---|---|---|---|
| #1 | Ali Khan | 0.88 | Matched: python, django, sql — Strong match |
| #2 | Priya Singh | 0.77 | Matched: django, python — Strong match |
| #3 | Sara Ahmed | 0.30 | Matched: sql — Weak match |

---

## 🧠 Scoring Formula
Final Score =

Semantic Similarity × 0.50

Skills Match      × 0.30
Experience Score  × 0.20
India Bonus       × 0.10


---

Built with ❤️ for India Runs Hackathon 2026