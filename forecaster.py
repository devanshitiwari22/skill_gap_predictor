import pandas as pd

SKILL_MARKET_INDEX = {
    "ai agents": {"base": 88, "growth_rate": 0.45, "category": "AI / Autonomous Systems"},
    "rag & vector dbs": {"base": 85, "growth_rate": 0.40, "category": "AI Architecture"},
    "llmops": {"base": 82, "growth_rate": 0.35, "category": "DevOps / MLOps"},
    "docker": {"base": 80, "growth_rate": 0.22, "category": "Cloud & Infrastructure"},
    "kubernetes": {"base": 78, "growth_rate": 0.24, "category": "Cloud & Infrastructure"},
    "fastapi": {"base": 75, "growth_rate": 0.26, "category": "Backend Engineering"},
    "deep learning": {"base": 80, "growth_rate": 0.20, "category": "AI / ML"},
    "machine learning": {"base": 85, "growth_rate": 0.12, "category": "AI / ML"},
    "nlp": {"base": 78, "growth_rate": 0.22, "category": "AI / ML"},
    "python": {"base": 95, "growth_rate": 0.08, "category": "Programming Languages"},
    "sql": {"base": 92, "growth_rate": 0.06, "category": "Data Management"},
    "git": {"base": 90, "growth_rate": 0.05, "category": "Core Tools"},
    "react": {"base": 84, "growth_rate": 0.09, "category": "Web Technologies"},
    "power bi": {"base": 78, "growth_rate": 0.10, "category": "Business Intelligence"},
    "excel": {"base": 70, "growth_rate": -0.04, "category": "Legacy Analytics"},
    "php": {"base": 45, "growth_rate": -0.08, "category": "Web Technologies"}
}

def project_skill_demand(skill_name, current_year=2026, passing_year=2027):
    skill = skill_name.lower().strip()
    data = SKILL_MARKET_INDEX.get(skill, {"base": 50, "growth_rate": 0.05, "category": "General Tech"})
    
    delta_t = max(0, passing_year - current_year)
    base = data["base"]
    r = data["growth_rate"]
    
    future_score = round(min(100.0, base * ((1 + r) ** delta_t)), 1)
    
    if r >= 0.25:
        trend_status = "🚀 Exponential Surge"
        risk_level = "Zero Risk (Prime Alpha)"
    elif r >= 0.05:
        trend_status = "📈 Stable Industry Need"
        risk_level = "Low Risk (Core Pillar)"
    else:
        trend_status = "⚠️ Stagnant / Automating"
        risk_level = "High Obsolescence Risk"
        
    return {
        "skill": skill.title(),
        "current_demand": base,
        "future_demand": future_score,
        "growth_pct": round(r * 100, 1),
        "status": trend_status,
        "risk": risk_level,
        "category": data["category"]
    }

def get_future_trend_series(skill_name, start_year=2023, end_year=2030):
    skill = skill_name.lower().strip()
    data = SKILL_MARKET_INDEX.get(skill, {"base": 50, "growth_rate": 0.05})
    
    years = list(range(start_year, end_year + 1))
    scores = []
    base_val = data["base"]
    r = data["growth_rate"]
    
    for y in years:
        diff = y - 2026
        val = min(100.0, max(10.0, base_val * ((1 + r) ** diff)))
        scores.append(round(val, 1))
        
    return pd.DataFrame({"Year": years, "Demand Index": scores, "Skill": skill.title()})