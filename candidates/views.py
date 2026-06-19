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
        jd = request.POST.get('jd', '').strip()

        # Error Check 1: Empty JD
        if not jd:
            return render(request, 'index.html', {
                'error': 'Please enter a Job Description!'
            })

        # Error Check 2: JD too short
        if len(jd) < 10:
            return render(request, 'index.html', {
                'error': 'Job Description is too short. Please add more details!'
            })

        try:
            # CSV upload check
            if 'csv_file' in request.FILES:
                csv_file = request.FILES['csv_file']

                # Error Check 3: File size
                if csv_file.size > 5 * 1024 * 1024:  # 5MB limit
                    return render(request, 'index.html', {
                        'error': 'CSV file too large. Max size is 5MB!'
                    })

                df = pd.read_csv(csv_file)

                # Error Check 4: Required columns
                required_cols = ['name', 'resume_text']
                missing_cols = [c for c in required_cols
                               if c not in df.columns]
                if missing_cols:
                    return render(request, 'index.html', {
                        'error': f'CSV missing required columns: {missing_cols}'
                    })

            else:
                csv_path = os.path.join('data', 'candidates.csv')
                df = pd.read_csv(csv_path)

            # Error Check 5: Empty dataset
            if len(df) == 0:
                return render(request, 'index.html', {
                    'error': 'Dataset is empty. Please upload a valid CSV!'
                })

            # AI ranking karo
            ranked = rank_candidates(jd, df)
            jd_info = parse_jd(jd)
            top10 = ranked.head(10).to_dict('records')

            return render(request, 'results.html', {
                'results': top10,
                'jd': jd,
                'total': len(ranked),
                'jd_info': jd_info
            })

        except Exception as e:
            return render(request, 'index.html', {
                'error': f'Something went wrong: {str(e)}'
            })

    return render(request, 'index.html')


def download_results(request):
    if request.method == 'POST':
        jd = request.POST.get('jd', '').strip()

        if not jd:
            return render(request, 'index.html', {
                'error': 'Please enter a Job Description!'
            })

        try:
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

        except Exception as e:
            return render(request, 'index.html', {
                'error': f'Download failed: {str(e)}'
            })

    return render(request, 'index.html')