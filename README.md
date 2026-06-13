# 🇮🇳 India Runs — AI Candidate Ranker

An AI-powered candidate ranking system built for the India Runs Hackathon by Redrob AI.

---

## What it does
Recruiters paste a Job Description — the AI finds and ranks 
the best matching candidates instantly.

---

## How it works
1. Job Description is converted to embeddings using BERT model
2. Each candidate resume is also converted to embeddings
3. Cosine similarity is calculated between JD and resumes
4. Skills matching score is calculated separately
5. Final score = Semantic (60%) + Skills Match (40%)
6. Candidates are ranked — best match first

---

## Tech Stack
- Python 3.10+
- Django 4.2
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn
- Pandas, NumPy

---

## Setup Instructions
```bash
# 1. Clone the repo
git clone https://github.com/TUMHARA_USERNAME/india-runs-candidate-ranker.git
cd india-runs-candidate-ranker

# 2. Virtual environment banao
python -m venv venv
venv\Scripts\activate

# 3. Dependencies install karo
pip install -r requirements.txt

# 4. Server run karo
python manage.py runserver
```

## Usage
1. Open `http://127.0.0.1:8000`
2. Paste any Job Description
3. Click "Find Best Candidates"
4. View ranked candidates with scores

---

## Project Structure
india_runs/

├── candidates/

│   ├── ai_engine.py     ← AI ranking logic

│   ├── views.py         ← Request handling

│   ├── urls.py          ← URL routing

│   └── templates/

│       ├── index.html   ← Input form

│       └── results.html ← Rankings display

├── data/

│   └── candidates.csv   ← Dataset

└── manage.py

## Results
| Rank | Candidate | Score |
|------|-----------|-------|
| #1 | Best Python Django match | 0.88 |
| #2 | Second best match | 0.77 |

---

Built with ❤️ for India Runs Hackathon 2026