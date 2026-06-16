from django.shortcuts import render
from django.http import HttpResponse
from .ai_engine import rank_candidates, parse_jd
import pandas as pd
import csv
import os

def index(request):
    return render(request, 'index.html')

def get_rankings(request):
    if request.method == 'POST':
        jd = request.POST.get('jd', '')

        if 'csv_file' in request.FILES:
            df = pd.read_csv(request.FILES['csv_file'])
        else:
            df = pd.read_csv(os.path.join('data', 'candidates.csv'))

        ranked = rank_candidates(jd, df)
        jd_info = parse_jd(jd)
        top10 = ranked.head(10).to_dict('records')

        return render(request, 'results.html', {
            'results': top10,
            'jd': jd,
            'total': len(ranked),
            'jd_info': jd_info
        })

    return render(request, 'index.html')

def download_results(request):
    if request.method == 'POST':
        jd = request.POST.get('jd', '')

        if 'csv_file' in request.FILES:
            df = pd.read_csv(request.FILES['csv_file'])
        else:
            df = pd.read_csv(os.path.join('data', 'candidates.csv'))

        ranked = rank_candidates(jd, df)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = \
            'attachment; filename="ranked_candidates.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Rank', 'Name', 'Final Score',
            'Semantic Score', 'Skills Score',
            'Experience Score', 'Behavioral Score',
            'Explanation'
        ])
        for _, row in ranked.iterrows():
            writer.writerow([
                row['rank'],
                row['name'],
                row['final_score'],
                row['semantic_score'],
                row['skills_score'],
                row['experience_score'],
                row['behavioral_score'],
                row['explanation']
            ])
        return response

    return render(request, 'index.html')