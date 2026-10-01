import re
from pypdf import PdfReader

SKILL_VOCABULARY = {
    "python", "java", "c++", "c", "c#", "r", "sql", "nosql", "mongodb", "postgresql", "mysql",
    "html", "css", "javascript", "typescript", "react", "angular", "vue", "nodejs", "express",
    "flask", "django", "fastapi", "spring boot", "rest api", "graphql",
    "machine learning", "deep learning", "nlp", "computer vision", "opencv", "pytorch", "tensorflow",
    "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "power bi", "tableau", "excel", "statistics", "data analysis", "exploratory data analysis",
    "docker", "kubernetes", "aws", "azure", "gcp", "linux", "bash", "git", "github", "ci/cd",
    "terraform", "kafka", "spark", "hadoop", "big data", "rag", "llmops", "ai agents"
}

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

def parse_skills_from_text(raw_text):
    if not raw_text:
        return []
    
    clean_text = raw_text.lower()
    clean_text = clean_text.replace("c++", "cpp").replace("c#", "csharp")
    clean_text = re.sub(r'[^a-zA-Z0-9\s,]', ' ', clean_text)
    
    found_skills = set()
    for skill in SKILL_VOCABULARY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, clean_text):
            found_skills.add(skill)
            
    raw_tokens = [s.strip().lower() for s in re.split(r'[,|\n]', raw_text) if s.strip()]
    for token in raw_tokens:
        if len(token) <= 25:
            found_skills.add(token)
            
    return sorted(list(found_skills))

def extract_skills_from_jd(jd_text):
    return parse_skills_from_text(jd_text)