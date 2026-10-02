import streamlit as st
import datetime
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from database import (
    init_db, get_roles, save_assessment_record, get_all_history, 
    get_tpo_analytics, add_user, login_user
)
from parser_utils import extract_text_from_pdf, parse_skills_from_text, extract_skills_from_jd
from forecaster import project_skill_demand, get_future_trend_series
from analyzer import calculate_urgency, predict_placement_metrics, analyze_skill_gap, generate_60_day_roadmap

def get_clean_name(username):
    if not username:
        return "Candidate"
    name_part = username.split("@")[0]
    clean = re.sub(r'\d+', '', name_part).replace('.', ' ').replace('_', ' ').replace('-', ' ').strip().title()
    return clean if clean else "Candidate"

def generate_dynamic_forecast_data(skills, scenario_multiplier=1.0):
    baseline_profiles = {
        "AI Agents": {"base": 14, "growth": 14.5, "cap": 99},
        "RAG & Vector DBs": {"base": 12, "growth": 13.8, "cap": 98},
        "LLMOps": {"base": 10, "growth": 14.0, "cap": 97},
        "Docker": {"base": 34, "growth": 9.2, "cap": 94},
        "Kubernetes": {"base": 30, "growth": 9.6, "cap": 93},
        "Python": {"base": 65, "growth": 4.8, "cap": 98},
        "SQL": {"base": 70, "growth": 3.4, "cap": 94},
        "Power BI": {"base": 48, "growth": 7.2, "cap": 93},
        "Data Cleaning": {"base": 58, "growth": 5.4, "cap": 94},
        "Statistics": {"base": 55, "growth": 5.2, "cap": 92},
        "OpenCV": {"base": 42, "growth": 6.8, "cap": 88},
        "FastAPI": {"base": 28, "growth": 9.8, "cap": 92},
        "React": {"base": 56, "growth": 4.2, "cap": 86},
        "Excel": {"base": 68, "growth": -3.8, "cap": 38},
        "PHP": {"base": 52, "growth": -4.6, "cap": 24}
    }
    
    years = list(range(2023, 2031))
    records = []
    
    for s in skills:
        prof = baseline_profiles.get(s, {"base": 45, "growth": 6.0, "cap": 90})
        b = prof["base"]
        g = prof["growth"] * scenario_multiplier
        cap = prof["cap"]
        
        for idx, y in enumerate(years):
            if g >= 0:
                val = b + (cap - b) * (1 / (1 + np.exp(-0.75 * (idx - 2.8 * scenario_multiplier))))
            else:
                val = max(cap, b + g * idx)
            val = round(float(np.clip(val, 10, 100)), 1)
            records.append({"Year": y, "Demand Index": val, "Skill": s})
            
    return pd.DataFrame(records)

init_db()

st.set_page_config(
    page_title="AI Future Skill Gap Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"], .stApp {
    background-color: #f8f6ff !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.16) 0%, transparent 55%),
        linear-gradient(rgba(124, 58, 237, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.05) 1px, transparent 1px) !important;
    background-size: 100% 100%, 28px 28px, 28px 28px !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #1e1b4b !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
}

p, span, div, li {
    color: #2e1065;
    font-weight: 600;
}

.stTextInput label, .stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label, .stFileUploader label, .stMultiSelect label {
    color: #1e1b4b !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
}

.feature-card-3d {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    text-align: center !important;
    box-shadow: 
        0 14px 30px -4px rgba(124, 58, 237, 0.14),
        0 4px 10px rgba(15, 23, 42, 0.04),
        inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.feature-card-3d:hover {
    transform: translateY(-4px) !important;
    border-color: #a78bfa !important;
    box-shadow: 
        0 22px 42px -6px rgba(124, 58, 237, 0.22),
        0 8px 18px -2px rgba(15, 23, 42, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
}

div[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 16px !important;
    box-shadow: 
        0 16px 36px -6px rgba(124, 58, 237, 0.16),
        0 6px 14px -2px rgba(15, 23, 42, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
    padding: 26px 30px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stForm"]:hover {
    transform: translateY(-4px) !important;
    border-color: #a78bfa !important;
    box-shadow: 
        0 24px 48px -8px rgba(124, 58, 237, 0.24),
        0 10px 20px -4px rgba(15, 23, 42, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
}

.gamertag-banner {
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    border: 2px solid #8b5cf6 !important;
    border-radius: 16px !important;
    padding: 22px 28px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    box-shadow: 
        0 18px 38px -6px rgba(109, 40, 217, 0.45),
        0 6px 14px rgba(0, 0, 0, 0.08),
        inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
    margin-bottom: 24px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.gamertag-banner:hover {
    transform: translateY(-3px) !important;
    box-shadow: 
        0 24px 48px -6px rgba(109, 40, 217, 0.55),
        0 8px 18px rgba(0, 0, 0, 0.12),
        inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
}

.gamertag-banner * {
    color: #ffffff !important;
}

.gamertag-banner .tag-cockpit {
    color: #e9d5ff !important;
    font-size: 11.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}

.gamertag-banner .banner-name {
    color: #ffffff !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    letter-spacing: -0.2px !important;
}

.gamertag-banner .tag-subtext {
    color: #f5f3ff !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    margin-top: 6px !important;
}

.gamertag-banner .tag-highlight {
    background: rgba(255, 255, 255, 0.18) !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    padding: 3px 10px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    font-size: 12.5px !important;
}

.gamertag-banner .tag-streak-box {
    background: rgba(255, 255, 255, 0.14) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.35) !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12) !important;
    text-align: right !important;
}

.gamertag-banner .tag-streak-label {
    color: #e9d5ff !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
}

.gamertag-banner .tag-gold {
    color: #ffffff !important;
    font-size: 22px !important;
    font-weight: 800 !important;
}

div[data-testid="stTabs"] > div:first-child {
    background-color: #ffffff !important;
    border: 2px solid #c4b5fd !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    box-shadow: 
        0 14px 30px -4px rgba(124, 58, 237, 0.14),
        0 4px 10px rgba(15, 23, 42, 0.04),
        inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
    margin-bottom: 24px !important;
    display: block !important;
}

div[data-testid="stTabs"] [role="tablist"],
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 12px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
}

div[data-testid="stTabs"] [role="tab"],
div[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: #f5f3ff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 10px !important;
    padding: 9px 20px !important;
    color: #4c1d95 !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    box-shadow: 0 2px 6px rgba(124, 58, 237, 0.06) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stTabs"] [role="tab"]:hover {
    background-color: #ede9fe !important;
    border-color: #c4b5fd !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(124, 58, 237, 0.16) !important;
}

div[data-testid="stTabs"] [aria-selected="true"],
div[data-testid="stTabs"] [aria-selected="true"] *,
div[data-testid="stTabs"] [aria-selected="true"] p,
div[data-testid="stTabs"] [aria-selected="true"] span,
div[data-testid="stTabs"] [aria-selected="true"] div {
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-color: #6d28d9 !important;
    box-shadow: 
        0 6px 16px rgba(109, 40, 217, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 16px !important;
    box-shadow: 
        0 14px 34px -6px rgba(124, 58, 237, 0.14),
        0 6px 14px -3px rgba(15, 23, 42, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
    padding: 24px 28px !important;
    margin-bottom: 24px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px) !important;
    border-color: #a78bfa !important;
    box-shadow: 
        0 22px 46px -8px rgba(124, 58, 237, 0.22),
        0 10px 20px -4px rgba(15, 23, 42, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
}

button[kind="primary"], 
.stButton > button[type="primary"],
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 15.5px !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 28px !important;
    box-shadow: 
        0 8px 22px rgba(109, 40, 217, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button[kind="primary"] *, .stFormSubmitButton > button * {
    color: #ffffff !important;
    font-weight: 800 !important;
}

button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-3px) scale(1.008) !important;
    box-shadow: 
        0 14px 30px rgba(109, 40, 217, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
}

.stButton > button {
    background-color: #ffffff !important;
    color: #4c1d95 !important;
    font-weight: 700 !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px rgba(124, 58, 237, 0.08) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #f5f3ff !important;
    border-color: #7c3aed !important;
    color: #6d28d9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(124, 58, 237, 0.15) !important;
}

.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #1e1b4b !important;
    font-weight: 600 !important;
    border: 1.5px solid #a78bfa !important;
    border-radius: 9px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.03) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25) !important;
}

span[data-baseweb="tag"] {
    background-color: #ede9fe !important;
    border: 1px solid #7c3aed !important;
    border-radius: 6px !important;
}
span[data-baseweb="tag"] span {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}

div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 
        0 8px 20px rgba(124, 58, 237, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 26px rgba(124, 58, 237, 0.16) !important;
}

div[data-testid="stMetricLabel"] {
    color: #4c1d95 !important;
    font-size: 13.5px !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
    color: #6d28d9 !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 2px solid #ddd6fe !important;
}
</style>
""", unsafe_allow_html=True)

def get_interview_questions(missing_skills):
    question_bank = {
        "docker": ["How do Docker multi-stage builds optimize production image size?", "Explain container bridge network isolation principles."],
        "kubernetes": ["What is the difference between a Pod, Deployment, and StatefulSet?", "How does Horizontal Pod Autoscaler evaluate scaling metrics?"],
        "pytorch": ["Explain the autograd engine and dynamic computational graph mechanics in PyTorch.", "How do you mitigate vanishing gradients in deep networks?"],
        "nlp": ["Explain the self-attention mechanism in modern Transformer architectures.", "Why is cosine similarity preferred over Euclidean distance for text embeddings?"],
        "sql": ["How does B-Tree indexing accelerate query lookup operations?", "Write an analytical query to evaluate top percentiles using window functions."],
        "git": ["Explain git rebase versus git merge under conflict scenarios.", "What is git cherry-pick and what are its standard enterprise use-cases?"],
        "ai agents": ["Explain LangGraph cyclic state execution versus linear chain pipelines.", "How do tool-calling function interfaces operate within autonomous LLM agents?"]
    }
    curated = []
    for skill in missing_skills:
        s_lower = skill.lower().strip()
        for key in question_bank:
            if key in s_lower:
                for q in question_bank[key]:
                    curated.append((skill.title(), q))
                break
    return curated[:4]

def render_urgency_badge(text, color_type):
    palette = {
        "danger": ("#fee2e2", "#991b1b", "#ef4444", "CRITICAL"),
        "warning": ("#fef3c7", "#92400e", "#f59e0b", "PRIORITY"),
        "info": ("#ede9fe", "#5b21b6", "#8b5cf6", "MODERATE"),
        "success": ("#dcfce7", "#166534", "#22c55e", "OPTIMAL")
    }
    bg, fg, border, label = palette.get(color_type, ("#ede9fe", "#5b21b6", "#8b5cf6", "STANDARD"))
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; background-color: {bg}; color: {fg}; border: 1.5px solid {border}; padding: 5px 14px; border-radius: 8px; font-weight: 800; font-size: 12.5px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
        <span style="font-weight: 800; letter-spacing: 0.5px;">[{label}]</span> <span>{text}</span>
    </div>
    """
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_type = ""

if "selected_skills_list" not in st.session_state:
    st.session_state.selected_skills_list = [
        "Data Cleaning", "Excel", "Exploratory Data Analysis", "Power BI", "SQL", "Statistics", "OpenCV"
    ]

if not st.session_state.logged_in:
    st.write("")
    st.markdown("""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 12px;">
            <span style="background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #ffffff; padding: 6px 20px; border-radius: 6px; font-size: 12px; font-weight: 800; letter-spacing: 1px; box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);">
                INSTITUTIONAL CAREER READINESS MATRIX
            </span>
        </div>
        <h1 style="text-align: center; font-size: 2.5rem; font-weight: 800; color: #1e1b4b; margin-bottom: 6px;">
            Future Skill Gap Predictor
        </h1>
    """, unsafe_allow_html=True)
    st.write("")
    col_l1, col_l2, col_l3 = st.columns([1, 1.8, 1])
    with col_l2:
        auth_tab = st.radio("Access Portal:", ["Sign In", "Create New Account"], horizontal=True)
        st.write("")
        
        if auth_tab == "Sign In":
            with st.form("login_form"):
                st.markdown("<h4 style='color: #1e1b4b; font-weight: 800; margin-bottom: 18px;'>Institutional Sign In</h4>", unsafe_allow_html=True)
                username = st.text_input("Institutional Email / Username", placeholder="e.g. devanshi.tiwari2206@gmail.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Access Workspace", use_container_width=True, type="primary")
                
                if submit_login:
                    if username and password:
                        user_type = login_user(username, password)
                        if user_type:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_type = user_type
                            st.success(f"Welcome, {get_clean_name(username)}")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please verify your details.")
                    else:
                        st.warning("Please provide both email and password.")
                        
        else:
            with st.form("signup_form"):
                st.markdown("<h4 style='color: #1e1b4b; font-weight: 800; margin-bottom: 18px;'>Register Identity</h4>", unsafe_allow_html=True)
                new_user = st.text_input("Institutional Email / Username", placeholder="e.g. your_name or email")
                new_pass = st.text_input("Set Password", type="password", placeholder="••••••••")
                account_type = st.selectbox("Role Identity:", ["Student", "TPO / Faculty Admin"])
                submit_signup = st.form_submit_button("Register Account", use_container_width=True, type="primary")
                
                if submit_signup:
                    if new_user and new_pass:
                        if add_user(new_user, new_pass, account_type):
                            st.success("Registration complete. Switch to 'Sign In' to access.")
                        else:
                            st.error("Account already exists with this username.")
                    else:
                        st.warning("All fields are mandatory.")
    st.stop()

display_name = get_clean_name(st.session_state.username)
with st.sidebar:
    st.markdown("""
        <div style="padding: 6px 0;">
            <span style="background: linear-gradient(135deg, #7c3aed, #4c1d95); color: #ffffff; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 800;">
                VERIFIED SESSION
            </span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"### {display_name}")
    st.markdown(f"Role: **{st.session_state.user_type}**")
    
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_type = ""
        st.rerun()
        
    st.divider()
    if st.session_state.user_type == "TPO / Faculty Admin":
        portal_mode = st.radio("Workspace View:", ["TPO & Placement Cell (Admin)", "Student / Candidate View"])
    else:
        portal_mode = "Student / Candidate View"
        
    st.divider()
    st.markdown("**Pre-Configured Stacks:**")
    if st.button("Load Data Analytics Stack"):
        st.session_state.selected_skills_list = [
            "Data Cleaning", "Excel", "Exploratory Data Analysis", "Power BI", "SQL", "Statistics", "OpenCV"
        ]
        st.rerun()
    if st.button("Load AI/ML Student Stack"):
        st.session_state.selected_skills_list = [
            "Python", "SQL", "Pandas", "Scikit-Learn", "Machine Learning", "Deep Learning", "Git"
        ]
        st.rerun()
    if st.button("Load Full-Stack Student Stack"):
        st.session_state.selected_skills_list = [
            "HTML", "CSS", "Javascript", "React", "NodeJS", "MongoDB", "Git"
        ]
        st.rerun()
if portal_mode in ["Student / Candidate View", "👤 Student / Candidate View"]:
    roles_df = get_roles()
    role_dict = dict(zip(roles_df['role_name'], roles_df['required_skills']))
    ctc_dict = dict(zip(roles_df['role_name'], roles_df['avg_ctc_lpa']))
    st.markdown(f"""
    <div class="gamertag-banner">
        <div>
            <div class="tag-cockpit">
                CANDIDATE ASSESSMENT PORTAL
            </div>
            <div class="banner-name">
                Candidate Profile: {display_name}
            </div>
            <div class="tag-subtext">
                Current Tier: <span class="tag-highlight">Level 3: Tech Apprentice</span> &nbsp;|&nbsp; 
                Target Trajectory: <span class="tag-highlight">Product Tier (10-18 LPA)</span>
    </div>
            <div class="tag-streak-box">
                <div class="tag-streak-label">
                    SESSION STREAK
            </div>
            <div class="tag-gold">
                04 Days Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    tab_eval, tab_3d, tab_forecast, tab_history = st.tabs([
        "Skill Assessment Lab", 
        "3D Vector Competency Space", 
        "2026–2030 Demand Forecasting", 
        "Assessment Records & Export"
    ])
    with tab_eval:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="background:#ede9fe; color:#6d28d9; padding:4px 10px; border-radius:6px; font-weight:800; font-size:12px;">SECTION 1</span>
                <h3 style="margin:0; color:#1e1b4b; font-weight:800;">Candidate Profile & Placement Horizon</h3>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            curr_year = datetime.datetime.now().year
            
            with c1:
                candidate_name = st.text_input("Full Name *", value=display_name)
            with c2:
                current_year = st.number_input("Current Year", min_value=2020, max_value=2035, value=curr_year)
            with c3:
                passing_year = st.number_input("Graduation / Passing Year", min_value=2020, max_value=2035, value=curr_year + 1)
                
            urgency_text, urgency_color = calculate_urgency(current_year, passing_year)
            st.markdown(f"**Placement Urgency Status:** {render_urgency_badge(urgency_text, urgency_color)}", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="background:#ede9fe; color:#6d28d9; padding:4px 10px; border-radius:6px; font-weight:800; font-size:12px;">SECTION 2</span>
                <h3 style="margin:0; color:#1e1b4b; font-weight:800;">Target Role & Market Intelligence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            role_mode = st.radio("Benchmark Mode:", ["Select Pre-Defined Role", "Paste Live Job Description (ATS Scanner)"], horizontal=True)
            
            target_role_name = ""
            benchmark_skills_str = ""
            
            if role_mode == "Select Pre-Defined Role":
                col_sel_role, col_intel = st.columns([1, 1], gap="large")
                with col_sel_role:
                    target_role_name = st.selectbox("Choose Target Track:", list(role_dict.keys()))
                    benchmark_skills_str = role_dict[target_role_name]
                    expected_ctc = ctc_dict.get(target_role_name, 10.0)
                
                with col_intel:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); border: 1.5px solid #c4b5fd; border-radius: 12px; padding: 14px 18px; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.08);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:800; color:#1e1b4b; font-size:13px; letter-spacing:0.4px;">MARKET INTEL: {target_role_name.upper()}</span>
                            <span style="background:#7c3aed; color:#fff; font-size:11px; padding:3px 8px; border-radius:4px; font-weight:700;">ACTIVE HIRING</span>
                        </div>
                        <div style="margin-top:8px; font-size:13px; color:#2e1065;">
                            • <strong>Average Fresher CTC:</strong> ₹{expected_ctc} LPA<br>
                            • <strong>2026 Hiring Velocity:</strong> High Growth (+28% YoY)<br>
                            • <strong>Required Core Skills:</strong> {len(benchmark_skills_str.split(','))} Technologies
                        </div>
                        <div style="margin-top:8px;">
                            {' '.join([f'<span style=\"background:#fff; color:#6d28d9; padding:2px 8px; border-radius:6px; font-size:11.5px; font-weight:800; margin-right:4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);\">{s.strip()}</span>' for s in benchmark_skills_str.split(',')[:6]])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                target_role_name = st.text_input("Job Role / Target Company:", placeholder="e.g. Associate AI Engineer at Microsoft")
                custom_jd_text = st.text_area("Paste Job Description (JD) text:", placeholder="Paste LinkedIn or campus drive JD content here...", height=110)
                if custom_jd_text.strip():
                    extracted_jd_skills = extract_skills_from_jd(custom_jd_text)
                    benchmark_skills_str = ", ".join(extracted_jd_skills)
                    st.success(f"ATS Scanner identified {len(extracted_jd_skills)} required skills from JD: " + ", ".join([f"`{s}`" for s in extracted_jd_skills]))
        
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="background:#ede9fe; color:#6d28d9; padding:4px 10px; border-radius:6px; font-weight:800; font-size:12px;">SECTION 3</span>
                <h3 style="margin:0; color:#1e1b4b; font-weight:800;">Candidate Competency Profile Input</h3>
            </div>
            """, unsafe_allow_html=True)
            
            input_type = st.radio("Choose Input Modality:", ["Interactive Skill Selector", "Upload Resume (PDF)", "Paste Custom Text"], horizontal=True)
            
            user_skills = []
            source_label = ""
            
            if input_type == "Interactive Skill Selector":
                source_label = "Interactive Skill Lab"
                st.caption("Select or verify technical competencies from the index:")
                
                POPULAR_TECH = [
                    "Data Cleaning", 
                    "Excel", 
                    "Exploratory Data Analysis", 
                    "Power BI", 
                    "SQL", 
                    "Statistics", 
                    "OpenCV",
                    "Python", 
                    "Machine Learning", 
                    "Deep Learning", 
                    "Computer Vision", 
                    "Pandas", 
                    "Scikit-Learn", 
                    "PyTorch", 
                    "TensorFlow", 
                    "NLP", 
                    "Docker", 
                    "Kubernetes", 
                    "Git", 
                    "FastAPI", 
                    "React", 
                    "NodeJS", 
                    "Tableau", 
                    "AWS", 
                    "AI Agents"
                ]
                
                selected_tech = st.multiselect(
                    "Validated Competencies:",
                    options=POPULAR_TECH + [s for s in st.session_state.selected_skills_list if s not in POPULAR_TECH],
                    default=st.session_state.selected_skills_list
                )
                st.session_state.selected_skills_list = selected_tech
                user_skills = [s.lower() for s in selected_tech]
                
                st.markdown(f"""
                <div style="background:#f5f3ff; border:1.5px dashed #8b5cf6; padding:8px 14px; border-radius:8px; margin-top:8px; display:inline-block;">
                    <strong style="color:#1e1b4b;">{len(user_skills)} Competencies Staged</strong> &nbsp;|&nbsp; 
                    <span style="color:#6d28d9; font-weight:700;">Ready for Model Evaluation</span>
                </div>
                """, unsafe_allow_html=True)
                
            elif input_type == "Upload Resume (PDF)":
                source_label = "Resume PDF"
                pdf_file = st.file_uploader("Upload candidate resume (PDF format)", type=["pdf"])
                if pdf_file:
                    raw_text = extract_text_from_pdf(pdf_file)
                    user_skills = parse_skills_from_text(raw_text)
                    st.success(f"Parsed {len(user_skills)} skills from resume!")
                    st.write(", ".join([f"`{s}`" for s in user_skills]))
            else:
                source_label = "Manual Text"
                manual_skills_text = st.text_area("Enter technical skills (comma separated):", placeholder="e.g. Data Cleaning, Excel, Power BI, SQL, Statistics, OpenCV", height=90)
                if manual_skills_text.strip():
                    user_skills = parse_skills_from_text(manual_skills_text)
                    
            st.write("")
            run_analysis = st.button("Execute Placement Diagnostic & Gap Analysis", type="primary", use_container_width=True)
        if run_analysis:
            if not candidate_name.strip():
                st.error("Please specify candidate name.")
            elif not benchmark_skills_str.strip():
                st.error("Target benchmark skills are missing.")
            elif not user_skills:
                st.error("No skills detected. Choose skills or upload resume.")
            else:
                results = analyze_skill_gap(user_skills, benchmark_skills_str, current_year, passing_year)
                placement_odds, predicted_ctc = predict_placement_metrics(results["final_score"], current_year, passing_year)
                
                st.session_state.last_analysis = {
                    "matched": results["matched"],
                    "missing": results["missing"],
                    "score": results["final_score"],
                    "role": target_role_name
                }
                
                save_assessment_record({
                    "name": candidate_name,
                    "current_year": current_year,
                    "passing_year": passing_year,
                    "role": target_role_name,
                    "source": source_label,
                    "score": results["final_score"],
                    "coverage": results["coverage"],
                    "placement_odds": placement_odds,
                    "predicted_ctc": predicted_ctc,
                    "urgency": urgency_text,
                    "missing": results["missing"]
                })
                
                if results["alert_type"] == "GREEN":
                    st.success(f"### STATUS: INDUSTRY & ROLE READY\nCandidate profile demonstrates high alignment with target benchmarks.")
                elif results["alert_type"] == "YELLOW":
                    st.warning(f"### STATUS: MODERATE COMPETENCY GAP\nCore fundamentals verified; specialized production tools require bridging.")
                else:
                    st.error(f"### STATUS: CRITICAL SKILL GAP DETECTED\nHigh probability of preliminary filter risk; core architecture skills missing.")
                    
                col_g, col_m = st.columns([1, 1], gap="large")
                
                with col_g:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=results["final_score"],
                        title={'text': "Market Readiness Score (%)", 'font': {'color': '#1e1b4b', 'size': 18, 'family': 'Plus Jakarta Sans'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#4c1d95', 'tickwidth': 2},
                            'bar': {'color': "#7c3aed"},
                            'steps': [
                                {'range': [0, 50], 'color': "#fee2e2"},
                                {'range': [50, 75], 'color': "#fef3c7"},
                                {'range': [75, 100], 'color': "#dcfce7"}
                            ],
                            'threshold': {'line': {'color': "#1e1b4b", 'width': 4}, 'thickness': 0.75, 'value': results["final_score"]}
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="#ffffff", 
                        plot_bgcolor="#ffffff",
                        font={'color': "#1e1b4b", 'family': 'Plus Jakarta Sans'},
                        height=290, 
                        margin=dict(t=35, b=5, l=30, r=30)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_m:
                    st.markdown("<h4 style='color:#1e1b4b; font-weight:800;'>Placement & Compensation Forecast</h4>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Placement Probability", f"{placement_odds}%")
                    m2.metric("Benchmark Match", f"{results['coverage']}%")
                    st.markdown(f"• **Estimated Compensation:** `<span style='color:#6d28d9; font-weight:800; font-size:15px;'>{predicted_ctc}</span>`", unsafe_allow_html=True)
                    st.markdown(f"• **Graduation Window:** **{max(0, passing_year - current_year)} year(s) remaining**")
                    st.markdown(f"• **ML Model Confidence:** **`{results['confidence']}%`**")
                    
                col_miss, col_match = st.columns(2)
                with col_miss:
                    st.markdown("<h4 style='color:#b91c1c; font-weight:800;'>Critical Competency Gaps (High Filter Risk)</h4>", unsafe_allow_html=True)
                    if results["missing"]:
                        for s in results["missing"]:
                            st.markdown(f"""
                            <div style="background-color: #fee2e2; border-left: 5px solid #dc2626; color: #991b1b; padding: 7px 14px; border-radius: 6px; font-weight: 800; margin-bottom: 6px; box-shadow: 0 2px 6px rgba(220,38,38,0.08);">
                                • {s.upper()}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("Zero gaps identified. Target role requirements verified.")
                
                with col_match:
                    st.markdown("<h4 style='color:#15803d; font-weight:800;'>Validated Competencies</h4>", unsafe_allow_html=True)
                    if results["matched"]:
                        for s in results["matched"]:
                            st.markdown(f"""
                            <div style="background-color: #dcfce7; border-left: 5px solid #16a34a; color: #166534; padding: 7px 14px; border-radius: 6px; font-weight: 800; margin-bottom: 6px; box-shadow: 0 2px 6px rgba(22,163,74,0.08);">
                                • {s.upper()}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No overlapping skills found with benchmark.")
                        
                st.divider()
                st.markdown("<h3 style='color:#1e1b4b; font-weight:800;'>Targeted Technical Interview Probes</h3>", unsafe_allow_html=True)
                interview_qs = get_interview_questions(results["missing"])
                if interview_qs:
                    for skill_name, question in interview_qs:
                        with st.expander(f"Technical Probe: {skill_name}"):
                            st.markdown(f"**Question:** *\"{question}\"*")
                            st.markdown("• *Guideline: Prepare an architectural explanation highlighting trade-offs.*")
                else:
                    st.info("No gaps found. Assessment recommends advanced system design evaluation.")

                # 60-Day Roadmap
                st.divider()
                st.markdown("<h3 style='color:#1e1b4b; font-weight:800;'>Structured 60-Day Competency Bridge Roadmap</h3>", unsafe_allow_html=True)
                roadmap_steps = generate_60_day_roadmap(results["missing"])
                if roadmap_steps:
                    for item in roadmap_steps:
                        with st.expander(f"{item['phase']} — Focus: {item['focus']}"):
                            st.markdown(f"**Key Deliverable:** {item['deliverable']}")
                            st.markdown(f"**Curated Curriculum Source:** `<span style='color:#6d28d9; font-weight:800;'>{item['resource']}</span>`", unsafe_allow_html=True)
                            
                st.divider()
                report_text = f"CANDIDATE: {candidate_name}\nROLE: {target_role_name}\nSCORE: {results['final_score']}%\nODDS: {placement_odds}%\nCTC: {predicted_ctc}\nMISSING: {', '.join(results['missing'])}\nMATCHED: {', '.join(results['matched'])}"
                st.download_button(
                    label="Export Assessment Report (TXT)",
                    data=report_text,
                    file_name=f"{candidate_name.replace(' ', '_')}_Report.txt",
                    mime="text/plain"
                )

    with tab_3d:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; color:#1e1b4b;'>3D Vector Competency Space (Multi-Axis Geometry)</h3>", unsafe_allow_html=True)
            st.caption("Spatial vector representation across Market Demand, Momentum & Candidate Match.")
            
            skills_to_plot = ["Python", "SQL", "Machine Learning", "Deep Learning", "Docker", "Kubernetes", "AI Agents", "FastAPI", "React", "Excel"]
            x_demand = [95, 92, 85, 80, 80, 78, 88, 75, 84, 70]
            y_growth = [8, 6, 12, 20, 22, 24, 45, 26, 9, -4]
            
            matched_cache = [s.lower() for s in st.session_state.get("last_analysis", {}).get("matched", [])]
            z_scores = []
            status_colors = []
            
            for s in skills_to_plot:
                if s.lower() in matched_cache:
                    z_scores.append(90)
                    status_colors.append("#5b21b6")
                else:
                    z_scores.append(25)
                    status_colors.append("#9333ea")
                    
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=x_demand,
                y=y_growth,
                z=z_scores,
                mode='markers+text',
                text=skills_to_plot,
                textposition='top center',
                marker=dict(
                    size=13,
                    color=status_colors,
                    opacity=0.92,
                    line=dict(color='#3b0764', width=2)
                )
            )])
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(
                        title='Market Demand (0–100)',
                        gridcolor='#ede9fe',
                        backgroundcolor='#faf5ff',
                        color='#1e1b4b'
                    ),
                    yaxis=dict(
                        title='YoY Growth (%)',
                        gridcolor='#ede9fe',
                        backgroundcolor='#faf5ff',
                        color='#1e1b4b'
                    ),
                    zaxis=dict(
                        title='Proficiency Index',
                        gridcolor='#ede9fe',
                        backgroundcolor='#faf5ff',
                        color='#1e1b4b'
                    ),
                    bgcolor='#ffffff'
                ),
                paper_bgcolor="#ffffff",
                font=dict(color="#1e1b4b", family="Plus Jakarta Sans"),
                height=550,
                margin=dict(l=0, r=0, b=0, t=20)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption("• Deep Royal Purple (#5b21b6) = Acquired Competencies | • Electric Purple (#9333ea) = Target Gap Vectors")

    with tab_forecast:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; color:#1e1b4b;'>2026–2030 Skill Demand Forecasting & Horizon Modeling</h3>", unsafe_allow_html=True)
            st.caption("Interactive multi-year predictive modeling with macroeconomic adoption scenarios and cohort horizon analytics.")
            
            f_col1, f_col2 = st.columns([1, 2], gap="large")
            
            with f_col1:
                forecast_skills = st.multiselect(
                    "Select Technologies to Forecast:",
                    options=[
                        "AI Agents", "RAG & Vector DBs", "LLMOps", "Docker", "Kubernetes", 
                        "Python", "SQL", "Power BI", "Data Cleaning", "Statistics", 
                        "OpenCV", "FastAPI", "React", "Excel", "PHP"
                    ],
                    default=["AI Agents", "Docker", "Python", "SQL", "Excel"]
                )
                
                forecast_horizon = st.slider("Target Horizon Year (Graduation):", 2026, 2030, 2029)
                
                market_scenario = st.selectbox(
                    "Market Adoption Scenario:",
                    [
                        "Standard Market Trajectory (Baseline)",
                        "Accelerated GenAI & Automation Surge (+25% Growth)",
                        "Enterprise Conservative Adoption (-15% Pace)"
                    ]
                )
                
                multiplier = 1.0
                if "Accelerated" in market_scenario:
                    multiplier = 1.25
                elif "Conservative" in market_scenario:
                    multiplier = 0.85
                    
            with f_col2:
                if forecast_skills:
                    forecast_df = generate_dynamic_forecast_data(forecast_skills, scenario_multiplier=multiplier)
                    
                    horizon_df = forecast_df[forecast_df["Year"] == forecast_horizon].sort_values(by="Demand Index", ascending=False)
                    base_df = forecast_df[forecast_df["Year"] == 2026].set_index("Skill")["Demand Index"]
                    
                    if not horizon_df.empty:
                        top_skill = horizon_df.iloc[0]["Skill"]
                        top_val = horizon_df.iloc[0]["Demand Index"]
                        avg_demand = round(horizon_df["Demand Index"].mean(), 1)
                        
                        
                        base_val = base_df.get(top_skill, 50.0)
                        growth_pct = round(((top_val - base_val) / max(1, base_val)) * 100, 1)
                        sign = "+" if growth_pct >= 0 else ""
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric(f"Top Skill in {forecast_horizon}", f"{top_skill}", f"{top_val}/100 Index")
                        k2.metric(f"Cohort Avg Demand ({forecast_horizon})", f"{avg_demand}%", f"Target Window")
                        k3.metric(f"Net Growth ({top_skill} vs 2026)", f"{sign}{growth_pct}%", f"{market_scenario.split('(')[0].strip()}")
                    
                    
                    fig_trend = px.line(
                        forecast_df, 
                        x="Year", 
                        y="Demand Index", 
                        color="Skill", 
                        markers=True,
                        template="plotly_white",
                        title=f"Predicted Trajectory (2023–2030) • Scenario: {market_scenario.split('(')[0].strip()}"
                    )
                    
                    
                    fig_trend.add_vrect(
                        x0=2026, x1=2030,
                        fillcolor="rgba(124, 58, 237, 0.06)", opacity=0.8,
                        layer="below", line_width=0,
                        annotation_text="Prediction Horizon Zone (2026–2030)", 
                        annotation_position="top left",
                        annotation_font_color="#6d28d9"
                    )
                    
                    
                    fig_trend.add_vline(
                        x=forecast_horizon, 
                        line_dash="dash", 
                        line_color="#7c3aed", 
                        line_width=2.5,
                        annotation_text=f"Selected Horizon: {forecast_horizon}",
                        annotation_font_color="#4c1d95"
                    )
                    
                    fig_trend.update_layout(
                        paper_bgcolor="#ffffff", 
                        plot_bgcolor="#ffffff",
                        height=420,
                        hovermode="x unified",
                        margin=dict(t=40, b=20, l=10, r=10),
                        font=dict(family="Plus Jakarta Sans", color="#1e1b4b"),
                        yaxis=dict(range=[10, 105], title="Market Demand Score (0–100)")
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
            
            if forecast_skills and not horizon_df.empty:
                st.markdown(f"#### Competency Demand Ranking for Graduation Year: `{forecast_horizon}`")
                
                table_rows = []
                for rank, (_, row) in enumerate(horizon_df.iterrows(), start=1):
                    s_name = row["Skill"]
                    idx_horizon = row["Demand Index"]
                    idx_2026 = base_df.get(s_name, idx_horizon)
                    diff = round(idx_horizon - idx_2026, 1)
                    
                    if diff > 10:
                        status = "High Growth Trajectory"
                    elif diff >= 0:
                        status = "Stable Production Utility"
                    else:
                        status = "Deprecating / Legacy Shift"
                        
                    sign = "+" if diff >= 0 else ""
                    table_rows.append({
                        "Rank": f"#{rank}",
                        "Technology": s_name,
                        "Current Index (2026)": f"{idx_2026}",
                        f"Predicted Index ({forecast_horizon})": f"{idx_horizon}/100",
                        "Projected Growth": f"{sign}{diff} pts",
                        "Recruitment Sentiment": status
                    })
                    
                rank_df = pd.DataFrame(table_rows)
                st.dataframe(rank_df, use_container_width=True, hide_index=True)

    
    with tab_history:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; color:#1e1b4b;'>Candidate Assessment History & Audit Log</h3>", unsafe_allow_html=True)
            history_df = get_all_history()
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No prior assessment records logged for this session yet.")

else:
    tpo_df = get_tpo_analytics()
    
    if tpo_df.empty:
        st.info("No candidate assessments logged yet.")
    else:
        with st.container(border=True):
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Batch Avg Score", f"{round(tpo_df['score'].mean(), 1)}%")
            kpi2.metric("Avg Placement Odds", f"{round(tpo_df['placement_odds'].mean(), 1)}%")
            kpi3.metric("Placement Ready (>=75%)", f"{len(tpo_df[tpo_df['score'] >= 75])}")
            kpi4.metric("High Gap Cohort (<50%)", f"{len(tpo_df[tpo_df['score'] < 50])}")
        
        with st.container(border=True):
            st.markdown("#### Institutional Skill Deficit Analysis")
            all_missing = []
            for row in tpo_df["missing_skills"].dropna():
                for item in row.split(","):
                    cleaned = item.strip().lower()
                    if cleaned:
                        all_missing.append(cleaned)
                        
            if all_missing:
                missing_series = pd.Series(all_missing).value_counts().reset_index()
                missing_series.columns = ["Skill", "Student Count"]
                fig_bar = px.bar(
                    missing_series.head(8), 
                    x="Skill", 
                    y="Student Count", 
                    text="Student Count", 
                    color="Student Count", 
                    template="plotly_white",
                    color_continuous_scale="Purples"
                )
                fig_bar.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
                st.plotly_chart(fig_bar, use_container_width=True)
            
        with st.container(border=True):
            st.markdown("#### Recruiter Shortlisting Matrix")
            score_threshold = st.slider("Minimum Readiness Threshold (%):", 40, 90, 70)
            shortlisted = tpo_df[tpo_df["score"] >= score_threshold][["candidate_name", "target_role", "score", "placement_odds", "predicted_ctc", "missing_skills"]]
            st.dataframe(shortlisted, use_container_width=True, hide_index=True)
