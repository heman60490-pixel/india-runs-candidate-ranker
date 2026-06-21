# 🇮🇳 India Runs — AI Candidate Ranker

## 🌐 Live Demo
**Try it now:** [india-runs-candidate-ranker.onrender.com](https://india-runs-candidate-ranker.onrender.com/)

*Note: Free tier hosting — first request may take 30-60 seconds to wake up.*

> An AI-powered candidate ranking system built for the **India Runs Hackathon 2026** by Redrob AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Django](https://img.shields.io/badge/Django-4.2-green?style=flat-square)
![AI](https://img.shields.io/badge/AI-BERT%20Embeddings-purple?style=flat-square)
![Hackathon](https://img.shields.io/badge/India%20Runs-2026-orange?style=flat-square)

---

## 🎯 Problem

Recruiters receive thousands of resumes — keyword filters don't work.
Context, skills, experience, and candidate activity need to be understood **together**.

## 💡 Solution

An intelligent ranking system that combines **5 signals** to rank candidates:

| Signal | What it does |
|---|---|
| 🧠 Semantic Understanding | BERT embeddings match meaning, not just keywords |
| 🛠️ Skills Matching | Exact skill overlap between JD and resume |
| 📅 Experience Scoring | Years of experience vs requirement |
| 📊 Behavioral Signals | Activity, profile completeness, application count |
| 🇮🇳 India Bonus | Recognizes Indian certifications (NPTEL, TCS iON, etc.) |

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐

│                    AI CANDIDATE RANKER                       │

│                     India Runs 2026                          │

└─────────────────────────────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│                      INPUT LAYER                             │

│                                                              │

│   ┌─────────────────┐         ┌─────────────────────┐       │

│   │  Job Description │         │  Candidates CSV      │       │

│   │  (Text Input)    │         │  resume_text         │       │

│   │                  │         │  last_active_days    │       │

│   │                  │         │  profile_score       │       │

│   │                  │         │  applications_count  │       │

│   └────────┬─────────┘         └──────────┬──────────┘       │

└────────────┼──────────────────────────────┼──────────────────┘

│                              │

▼                              ▼

┌─────────────────────────────────────────────────────────────┐

│                    PROCESSING LAYER                          │

│                                                              │

│   ┌──────────────────────────────────────────────────────┐  │

│   │              JD PARSER (Deep Understanding)           │  │

│   │   Role Level Detection │ Domain Detection             │  │

│   │   Skills Extraction    │ Experience Parsing           │  │

│   └──────────────────────────────────────────────────────┘  │

│                              │                               │

│   ┌──────────────┐  ┌────────┴───────┐  ┌────────────────┐  │

│   │   SEMANTIC   │  │    SKILLS      │  │  BEHAVIORAL    │  │

│   │   ENGINE     │  │    MATCHER     │  │  ANALYZER      │  │

│   │              │  │                │  │                │  │

│   │ BERT Model   │  │ Skill Overlap  │  │ Activity Score │  │

│   │ Embeddings   │  │ JD vs Resume   │  │ Profile Score  │  │

│   │ Cosine Sim   │  │ Exact Match    │  │ App Count      │  │

│   └──────┬───────┘  └───────┬────────┘  └───────┬────────┘  │

│          │                  │                    │            │

│          ▼                  ▼                    ▼            │

│   ┌──────────────────────────────────────────────────────┐  │

│   │              WEIGHTED SCORING ENGINE                  │  │

│   │                                                        │  │

│   │   Semantic Score  × 0.40  (40%)                       │  │

│   │ + Skills Score    × 0.25  (25%)                       │  │

│   │ + Experience      × 0.15  (15%)                       │  │

│   │ + Behavioral      × 0.20  (20%)                       │  │

│   │ + India Bonus     × 0.10  (max 10%)                   │  │

│   │                   ─────────────                        │  │

│   │                   FINAL SCORE                          │  │

│   └──────────────────────────────────────────────────────┘  │

└─────────────────────────────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│                      OUTPUT LAYER                            │

│                                                              │

│   ┌─────────────────┐         ┌─────────────────────┐       │

│   │  Ranked List     │         │  Explanation         │       │

│   │  #1 Best Match   │         │  "Matched: python,   │       │

│   │  #2 Second Best  │         │   django, sql"       │       │

│   │  #3 ...          │         │  "Strong match"      │       │

│   └─────────────────┘         └─────────────────────┘       │

│                                                              │

│   ┌─────────────────────────────────────────────────────┐   │

│   │              Django Web Interface                     │   │

│   │   index.html (Input) → views.py → results.html       │   │

│   └─────────────────────────────────────────────────────┘   │

└─────────────────────────────────────────────────────────────┘

---

## 🔢 Scoring Formula
Final Score =

Semantic Similarity  × 0.40   ← BERT Deep Understanding

Skills Match Score   × 0.25   ← Exact Skill Overlap
Experience Score     × 0.15   ← Years vs Required
Behavioral Score     × 0.20   ← Activity + Profile + Apps
India Bonus          × 0.10   ← Indian Certifications


---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Django 4.2 | Web framework |
| sentence-transformers | BERT embeddings (all-MiniLM-L6-v2) |
| scikit-learn | Cosine similarity |
| Pandas + NumPy | Data processing |

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/heman60490-pixel/india-runs-candidate-ranker.git
cd india-runs-candidate-ranker

# 2. Virtual environment banao
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Dependencies install karo
pip install -r requirements.txt

# 4. Server run karo
python manage.py runserver
```

Open `http://127.0.0.1:8000` in browser.

---

## 📊 Features

| Feature | Description |
|---|---|
| 🧠 Semantic Ranking | BERT-based meaning match — not just keywords |
| 🛠️ Skills Matching | Exact skill overlap detection |
| 📅 Experience Score | Years of experience vs requirement |
| 📊 Behavioral Signals | Activity, profile completeness, applications |
| 🇮🇳 India Bonus | Indian certification recognition |
| 🔍 Explainability | Why each candidate was ranked |
| 📂 CSV Upload | Upload your own candidate dataset |
| ⬇️ Download Results | Export ranked candidates as CSV |
| 🌐 Web Interface | Clean Django-powered UI |
| 🔎 JD Parser | Role level, domain, skills auto-detection |

---

## 📁 Project Structure
india_runs/

├── candidates/

│   ├── ai_engine.py        ← Core AI ranking logic

│   ├── views.py            ← Request handling

│   ├── urls.py             ← URL routing

│   └── templates/

│       ├── index.html      ← JD input form

│       └── results.html    ← Ranked results display

├── data/

│   ├── candidates.csv      ← Candidate dataset

│   └── ranked_output.csv   ← Generated ranked output

├── manage.py

├── requirements.txt

└── README.md

---

## 📈 Sample Results

| Rank | Candidate | Final Score | Signals | Explanation |
|---|---|---|---|---|
| #1 | Priya Singh | 0.915 | Semantic: 0.75, Skills: 1.0, Behavioral: 1.0 | Matched: django, rest api — Strong match |
| #2 | Ali Khan | 0.901 | Semantic: 0.79, Skills: 1.0, Behavioral: 0.98 | Matched: django, rest api — Strong match |
| #3 | Ravi Sharma | 0.423 | Semantic: 0.26, Skills: 0.0, Behavioral: 0.86 | Missing: django — Moderate match |
| #4 | Sara Ahmed | 0.374 | Semantic: 0.32, Skills: 0.0, Behavioral: 0.47 | Missing: django — Weak match |
| #5 | John Doe | 0.301 | Semantic: 0.27, Skills: 0.0, Behavioral: 0.20 | Missing: django, rest api — Weak match |

---

## ⚡ Scalability Test

Tested with **150 synthetic candidates** to verify batch processing performance:

## 🎯 What Makes This Different

- **Not just keyword matching** — BERT understands meaning and context
- **Behavioral signals** — active candidates ranked higher
- **India-specific** — recognizes Indian certifications and market
- **Explainable AI** — tells WHY each candidate was ranked
- **Full web app** — not just a script, a complete recruiter tool

---

## ⚠️ Note on Performance

This demo is hosted on Render's free tier (512MB RAM), 
which has two limitations:

1. **Cold start**: If inactive for 15+ minutes, the first 
   request takes 30-60 seconds to wake up the server.

2. **Slower inference**: Due to limited RAM, the BERT model 
   inference is slower than on standard hardware (20-30s 
   vs 5-10s locally).

**This is a hosting limitation, not a code limitation.** 
The system has been tested locally with 150 candidates 
in under 10 seconds (see Scalability Test section above). 
On production-grade infrastructure (4GB+ RAM), 
performance would match local benchmarks.

Built with ❤️ for India Runs Hackathon 2026