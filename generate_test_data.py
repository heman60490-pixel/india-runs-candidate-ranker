import pandas as pd
import random

names = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Gupta",
    "Vikram Singh", "Anjali Mehta", "Rohan Joshi", "Pooja Nair",
    "Arjun Reddy", "Kavya Iyer", "Mohammed Ali", "Sunita Rao",
    "Deepak Verma", "Ritika Bansal", "Suresh Menon", "Ananya Das",
    "Karan Malhotra", "Divya Krishnan", "Nikhil Bose", "Shreya Shah"
]

resumes = [
    "Python Django REST API developer {exp} years SQL PostgreSQL B.Tech",
    "Java Spring Boot developer {exp} years MySQL Microservices B.Tech",
    "React JavaScript frontend developer {exp} years HTML CSS M.Tech",
    "Machine Learning engineer {exp} years Python TensorFlow Pandas BCA",
    "Python Flask MongoDB developer {exp} years Docker AWS B.Tech",
    "Django Python PostgreSQL developer {exp} years Redis Celery MCA",
    "JavaScript React Node.js developer {exp} years MongoDB B.Sc",
    "Python Data Science {exp} years Pandas NumPy Matplotlib B.Tech",
    "Java Python developer {exp} years SQL Docker Kubernetes AWS M.Tech",
    "Django REST API Python developer {exp} years MySQL HTML CSS B.Tech",
]

data = []
for i in range(150):  # 150 candidates banayenge
    name = random.choice(names) + f" {i+1}"
    exp = random.randint(1, 8)
    resume = random.choice(resumes).format(exp=exp)

    data.append({
        'candidate_id': i + 1,
        'name': name,
        'resume_text': resume,
        'last_active_days': random.randint(1, 180),
        'profile_score': random.randint(40, 100),
        'applications_count': random.randint(1, 20)
    })

df = pd.DataFrame(data)
df.to_csv('data/test_large_dataset.csv', index=False)
print(f"✅ Test dataset ready: {len(df)} candidates!")
print(f"📁 Saved to: data/test_large_dataset.csv")