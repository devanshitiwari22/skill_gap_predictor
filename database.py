import sqlite3
import pandas as pd

DB_NAME = "skills.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT UNIQUE NOT NULL,
            required_skills TEXT NOT NULL,
            avg_ctc_lpa REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidate_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            current_year INTEGER NOT NULL,
            passing_year INTEGER NOT NULL,
            target_role TEXT NOT NULL,
            input_source TEXT NOT NULL,
            score REAL NOT NULL,
            coverage REAL NOT NULL,
            placement_odds REAL NOT NULL,
            predicted_ctc TEXT NOT NULL,
            urgency_level TEXT NOT NULL,
            missing_skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    roles_data = [
        ("AI / ML Engineer", "python, machine learning, deep learning, pytorch, tensorflow, nlp, sql, docker, git, opencv", 12.5),
        ("Data Scientist", "python, r, sql, statistics, machine learning, pandas, tableau, big data, deep learning", 11.0),
        ("Full Stack Developer", "javascript, react, nodejs, express, mongodb, sql, html, css, git, docker, rest api", 9.0),
        ("Data Analyst", "sql, excel, power bi, tableau, python, exploratory data analysis, statistics, data cleaning", 6.5),
        ("Cloud & DevOps Engineer", "linux, docker, kubernetes, aws, terraform, ci/cd, python, bash, git, azure", 11.5)
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO benchmark_roles (role_name, required_skills, avg_ctc_lpa)
        VALUES (?, ?, ?)
    ''', roles_data)
    
    conn.commit()
    conn.close()

def get_roles():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT role_name, required_skills, avg_ctc_lpa FROM benchmark_roles", conn)
    conn.close()
    return df

def save_assessment_record(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidate_assessments 
        (candidate_name, current_year, passing_year, target_role, input_source, score, coverage, placement_odds, predicted_ctc, urgency_level, missing_skills)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['name'], data['current_year'], data['passing_year'], 
        data['role'], data['source'], data['score'], data['coverage'],
        data['placement_odds'], data['predicted_ctc'], data['urgency'], 
        ", ".join(data['missing'])
    ))
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM candidate_assessments ORDER BY id DESC", conn)
    conn.close()
    return df

def get_tpo_analytics():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM candidate_assessments", conn)
    conn.close()
    return df
import hashlib

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def add_user(username, password, user_type="Student"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL
        )
    ''')
    try:
        cursor.execute('INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)',
                       (username, make_hashes(password), user_type))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL
        )
    ''')
    cursor.execute('SELECT user_type FROM users WHERE username = ? AND password = ?',
                   (username, make_hashes(password)))
    data = cursor.fetchone()
    conn.close()
    return data[0] if data else None