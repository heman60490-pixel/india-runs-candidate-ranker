from django.shortcuts import render
from .ai_engine import rank_candidates
import pandas as pd
import os

def index(request):
    return render(request, 'index.html')

def get_rankings(request):
    if request.method == 'POST':
        # Step 1: User ki JD lo
        jd = request.POST.get('jd', '')

        # Step 2: Dataset load karo
        csv_path = os.path.join('data', 'candidates.csv')
        df = pd.read_csv(csv_path)

        # Step 3: AI engine ko do — ranked list wapas aayegi
        ranked = rank_candidates(jd, df)

        # Step 4: Top 10 results lo
        top10 = ranked.head(10).to_dict('records')

        # Step 5: results.html ko do
        return render(request, 'results.html', {
            'results': top10,
            'jd': jd
        })

    return render(request, 'index.html')