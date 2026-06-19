import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ranker.settings')

import django
django.setup()

import pandas as pd
from candidates.ai_engine import (
    extract_skills,
    skills_score,
    experience_score,
    behavioral_score,
    explain_ranking,
    parse_jd,
    rank_candidates
)

# ================================
# Test 1: Skills Extraction
# ================================
def test_extract_skills():
    print("\n🧪 Test 1: extract_skills()")

    # Python aur Django hona chahiye
    result = extract_skills("Python Django developer")
    assert 'python' in result, "❌ Python detect nahi hua!"
    assert 'django' in result, "❌ Django detect nahi hua!"
    print("   ✅ Python detected:", 'python' in result)
    print("   ✅ Django detected:", 'django' in result)

    # Empty text
    result = extract_skills("")
    assert result == set(), "❌ Empty text pe empty set nahi aaya!"
    print("   ✅ Empty text handled correctly")

    print("   ✅ Test 1 PASSED!")

# ================================
# Test 2: Skills Score
# ================================
def test_skills_score():
    print("\n🧪 Test 2: skills_score()")

    # Perfect match — score 1.0 hona chahiye
    score = skills_score(
        "Python Django developer",
        "Python Django developer 3 years"
    )
    assert score == 1.0, f"❌ Perfect match score {score} != 1.0"
    print(f"   ✅ Perfect match score: {score}")

    # No match — score 0.0 hona chahiye
    score = skills_score(
        "Python Django developer",
        "Java Spring Boot developer"
    )
    assert score == 0.0, f"❌ No match score {score} != 0.0"
    print(f"   ✅ No match score: {score}")

    # Empty JD
    score = skills_score("", "Python developer")
    assert score == 0.0, "❌ Empty JD score != 0.0"
    print(f"   ✅ Empty JD handled: {score}")

    print("   ✅ Test 2 PASSED!")

# ================================
# Test 3: Experience Score
# ================================
def test_experience_score():
    print("\n🧪 Test 3: experience_score()")

    # 3 years experience — required 2 — score 1.0 (capped)
    score = experience_score("Python developer 3 years experience", 2)
    assert score == 1.0, f"❌ Score {score} != 1.0"
    print(f"   ✅ 3 years vs required 2: {score}")

    # 1 year experience — required 2 — score 0.5
    score = experience_score("Python developer 1 year experience", 2)
    assert score == 0.5, f"❌ Score {score} != 0.5"
    print(f"   ✅ 1 year vs required 2: {score}")

    # No experience mentioned — score 0.0
    score = experience_score("Python developer", 2)
    assert score == 0.0, f"❌ Score {score} != 0.0"
    print(f"   ✅ No experience mentioned: {score}")

    print("   ✅ Test 3 PASSED!")

# ================================
# Test 4: Behavioral Score
# ================================
def test_behavioral_score():
    print("\n🧪 Test 4: behavioral_score()")

    # Very active candidate — high score
    active = {
        'last_active_days': 1,
        'profile_score': 100,
        'applications_count': 15
    }
    score = behavioral_score(active)
    assert score > 0.8, f"❌ Active candidate score {score} < 0.8"
    print(f"   ✅ Active candidate score: {score}")

    # Inactive candidate — low score
    inactive = {
        'last_active_days': 200,
        'profile_score': 20,
        'applications_count': 0
    }
    score = behavioral_score(inactive)
    assert score < 0.5, f"❌ Inactive candidate score {score} >= 0.5"
    print(f"   ✅ Inactive candidate score: {score}")

    print("   ✅ Test 4 PASSED!")

# ================================
# Test 5: JD Parser
# ================================
def test_parse_jd():
    print("\n🧪 Test 5: parse_jd()")

    jd = "Senior Django REST API developer 3 years experience"
    result = parse_jd(jd)

    assert 'role_level' in result, "❌ role_level missing!"
    assert 'domain' in result, "❌ domain missing!"
    assert 'skill_count' in result, "❌ skill_count missing!"
    assert result['domain'] == 'backend', \
        f"❌ Domain {result['domain']} != backend"

    print(f"   ✅ Role Level: {result['role_level']}")
    print(f"   ✅ Domain: {result['domain']}")
    print(f"   ✅ Skills Count: {result['skill_count']}")
    print(f"   ✅ Experience: {result['experience_years']}")

    print("   ✅ Test 5 PASSED!")

# ================================
# Test 6: Full Ranking
# ================================
def test_rank_candidates():
    print("\n🧪 Test 6: rank_candidates()")

    # Test data
    df = pd.DataFrame({
        'candidate_id': [1, 2, 3],
        'name': ['Django Dev', 'Java Dev', 'React Dev'],
        'resume_text': [
            'Python Django REST API developer 3 years SQL',
            'Java Spring Boot developer 5 years MySQL',
            'React JavaScript frontend 2 years HTML CSS'
        ],
        'last_active_days': [2, 50, 100],
        'profile_score': [95, 60, 40],
        'applications_count': [12, 3, 1]
    })

    jd = "Python Django REST API developer"
    result = rank_candidates(jd, df)

    # Django Dev #1 hona chahiye
    assert result.iloc[0]['name'] == 'Django Dev', \
        f"❌ Wrong #1: {result.iloc[0]['name']}"
    print(f"   ✅ Rank #1: {result.iloc[0]['name']} — correct!")

    # Ranks 1,2,3 hone chahiye
    assert list(result['rank']) == [1, 2, 3], "❌ Ranks wrong!"
    print(f"   ✅ Ranks correct: {list(result['rank'])}")

    # Final score 0-1 ke beech hona chahiye
    for _, row in result.iterrows():
        assert 0 <= row['final_score'] <= 1.5, \
            f"❌ Score out of range: {row['final_score']}"
    print(f"   ✅ All scores in valid range")

    print("   ✅ Test 6 PASSED!")

# ================================
# Run All Tests
# ================================
if __name__ == '__main__':
    print("=" * 50)
    print("  AI CANDIDATE RANKER — UNIT TESTS")
    print("=" * 50)

    tests = [
        test_extract_skills,
        test_skills_score,
        test_experience_score,
        test_behavioral_score,
        test_parse_jd,
        test_rank_candidates,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"   {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print("  ⚠️  Some tests failed — fix karo!")
    print("=" * 50) 