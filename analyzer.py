import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("skill_evaluator_rf.pkl")

FUTURE_TECH = ["ai agents", "rag", "docker", "kubernetes", "fastapi", "llmops", "deep learning"]

def calculate_urgency(current_year, passing_year):
    years_left = passing_year - current_year
    if years_left <= 0:
        return "Immediate (Active Placement Season)", "danger"
    elif years_left == 1:
        return "High Priority (Pre-Final Year)", "warning"
    elif years_left == 2:
        return "Moderate (Preparation Phase)", "info"
    else:
        return "Low Urgency (Early Foundation)", "success"

def analyze_skill_gap(user_skills_list, benchmark_skills_raw, current_year=2026, passing_year=2027):
    benchmark_list = [s.strip().lower() for s in benchmark_skills_raw.split(',') if s.strip()]
    user_set = set([s.lower() for s in user_skills_list])
    benchmark_set = set(benchmark_list)

    matched = sorted(list(user_set.intersection(benchmark_set)))
    missing = sorted(list(benchmark_set.difference(user_set)))
    extra = sorted(list(user_set.difference(benchmark_set)))

    coverage = len(matched) / len(benchmark_list) if benchmark_list else 0.0

    doc_user = " ".join(user_skills_list)
    doc_bench = " ".join(benchmark_list)

    if not doc_user.strip():
        similarity = 0.0
    else:
        vectorizer = TfidfVectorizer().fit([doc_user, doc_bench])
        vecs = vectorizer.transform([doc_user, doc_bench])
        similarity = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])

    future_hits = sum(1 for s in user_skills_list if any(ft in s.lower() for ft in FUTURE_TECH))
    future_weight = min(1.0, future_hits / 3.0)

    years_left = max(0, passing_year - current_year)
    missing_count = len(missing)

    features = pd.DataFrame([{
        'cosine_sim': similarity,
        'coverage': coverage,
        'missing_count': missing_count,
        'years_left': years_left,
        'future_weight': future_weight
    }])

    pred_class = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    final_score = round(((0.6 * coverage) + (0.4 * similarity)) * 100, 1)
    coverage_pct = round(coverage * 100, 1)
    similarity_pct = round(similarity * 100, 1)

    if pred_class == 2:
        alert_type = "GREEN"
        alert_title = "🟢 GREEN ALERT: Industry & Role Ready!"
        alert_desc = "High competency match. Candidate profile qualifies for Day-1 shortlisting."
    elif pred_class == 1:
        alert_type = "YELLOW"
        alert_title = "🟡 AMBER ALERT: Moderate Competency Gap"
        alert_desc = "Strong fundamentals present, but missing key specialized industry tools."
    else:
        alert_type = "RED"
        alert_title = "🔴 RED ALERT: Critical Skill Gap Detected!"
        alert_desc = "High risk of resume filtering. Essential architecture and core skills are absent."

    return {
        "final_score": final_score,
        "coverage": coverage_pct,
        "similarity": similarity_pct,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "alert_type": alert_type,
        "alert_title": alert_title,
        "alert_desc": alert_desc,
        "pred_class": int(pred_class),
        "confidence": round(float(np.max(probs)) * 100, 1)
    }

def predict_placement_metrics(final_score, current_year, passing_year):
    years_left = max(0, passing_year - current_year)
    raw_odds = (final_score * 0.85) + (years_left * 4.5)
    placement_odds = min(98.5, max(15.0, round(raw_odds, 1)))

    if final_score >= 80:
        predicted_ctc = "10.0 – 18.0 LPA (Tier-1 Product / AI Core)"
    elif final_score >= 65:
        predicted_ctc = "6.5 – 10.0 LPA (Mid-Tier Tech / High Growth)"
    elif final_score >= 50:
        predicted_ctc = "4.5 – 6.5 LPA (Entry-Level Developer / Analyst)"
    else:
        predicted_ctc = "3.2 – 4.5 LPA (Service / Support Track)"

    return placement_odds, predicted_ctc

def generate_60_day_roadmap(missing_skills):
    if not missing_skills:
        return []

    return [
        {
            "phase": "Weeks 1–2: Foundational Architecture",
            "focus": missing_skills[0].title() if len(missing_skills) > 0 else "Core Concepts",
            "deliverable": "Understand syntax, architecture & configure developer environment.",
            "resource": "Official Documentation & FreeCodeCamp Deep Dives"
        },
        {
            "phase": "Weeks 3–4: Hands-On Mini Implementations",
            "focus": missing_skills[1].title() if len(missing_skills) > 1 else "API / Tool Integration",
            "deliverable": "Build standalone scripts or automated pipelines solving real edge cases.",
            "resource": "GitHub open-source repositories & Kaggle notebooks"
        },
        {
            "phase": "Weeks 5–6: Capstone Project Construction",
            "focus": f"Integration of {', '.join([s.title() for s in missing_skills[:2]])}",
            "deliverable": "Containerize with Docker, deploy live API, and publish repository with documentation.",
            "resource": "Architecture design patterns & system design blogs"
        },
        {
            "phase": "Weeks 7–8: Mock Assessments & ATS Tuning",
            "focus": "Interview Readiness & System Optimization",
            "deliverable": "Live project presentation demo + GitHub badge certification update.",
            "resource": "LeetCode / Tech Interview question handbooks"
        }
    ]