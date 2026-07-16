from flask import Flask, render_template, request, redirect, session, flash, send_from_directory
import joblib
import os
import sqlite3
import requests
import time
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

API_KEY = os.environ.get("NEWSAPI_KEY", "01005f5f946a4d8b9c232a8ee206c88d")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_KEY", ""))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- DATABASE ----------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    conn.commit()
    conn.close()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = None
vectorizer = None
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()

        hashed_password = generate_password_hash(password, method='pbkdf2')

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )

        conn.commit()
        conn.close()

        flash("Registered successfully!")
        return redirect('/login')

    return render_template('register.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        try:
            password_ok = user and check_password_hash(user[2], password)
        except ValueError:
            password_ok = False

        if password_ok:
            session['user'] = username
            return redirect('/')
        else:
            flash("Invalid login")

    return render_template('login.html')

# ---------------- HOME ----------------
@app.route('/')
def home():
    if 'user' not in session:
        return redirect('/login')
    return render_template("index.html")

# ---------------- NEWS ----------------
@app.route('/news')
def news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
        res = requests.get(url)
        data = res.json()

        articles = data.get('articles', [])

        return render_template("news.html", articles=articles)

    except Exception as e:
        return f"Error loading news: {str(e)}"

# ---------------- VERIFY (Websearch) ----------------
@app.route('/verify', methods=['POST'])
def verify():
    news = request.form.get('news', '').strip()

    if not news:
        return render_template("index.html", prediction=None, confidence=None, verification="Error: No news text provided for verification.")

    if not OPENROUTER_API_KEY:
        return render_template("index.html", prediction=None, confidence=None, verification="Error: OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")

    try:
        verification = call_openrouter_verification(news)
        return render_template("index.html", prediction=None, confidence=None, verification=verification)
    except Exception as e:
        error_str = str(e)
        print(f"Verification error: {error_str}")
        return render_template("index.html", prediction=None, confidence=None, verification=f"Verification error: {error_str}")


def call_openrouter_verification(news: str) -> str:
    live_context = fetch_live_news_context()

    prompt = (
        "You are a professional fact-checking assistant with access to the latest news context below. "
        "Your task is to verify whether the following news claim is real or fake using the provided current headlines and trusted sources. "
        "Base your answer on the live context first. If the live context clearly corroborates or refutes the claim, say so explicitly. "
        "If the live context is insufficient, say so honestly. "
        "Always provide: (1) Verdict, (2) Evidence from live context, (3) Confidence level.\n\n"
    )

    if live_context:
        prompt += "=== LATEST NEWS CONTEXT (from NewsAPI) ===\n" + live_context + "\n\n"

    prompt += "=== CLAIM TO VERIFY ===\n" + news + "\n\nAnswer:"

    models_to_try = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-26b-a4b-it:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openrouter/free",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512,
            )
            content = response.choices[0].message.content if response.choices else ""
            result = content.strip() or "No verification result returned by the model."
            if live_context:
                result += "\n\n(Verification based on live news context fetched from NewsAPI)"
            return result
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str:
                time.sleep(2)
                continue
            if "404" in error_str or "unavailable" in error_str.lower():
                print(f"Model {model_name} unavailable, skipping...")
                continue
            print(f"Model {model_name} failed: {error_str}")

    if last_error:
        raise last_error
    return "Verification failed: unable to reach AI model."


def fetch_live_news_context() -> str:
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=8&apiKey={API_KEY}"
        res = requests.get(url, timeout=10)
        data = res.json()
        articles = data.get('articles', [])
        if not articles:
            return ""
        lines = []
        for i, article in enumerate(articles[:8], 1):
            title = article.get('title', '')
            source = article.get('source', {}).get('name', '')
            if title:
                lines.append(f"{i}. {title} [{source}]")
        return "\n".join(lines)
    except Exception as e:
        print(f"Failed to fetch live news context: {e}")
        return ""

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return render_template("index.html", prediction=None, confidence=None, verification="Error: ML model not loaded. Run train_model.py first.")

    news = request.form['news']
    use_verification = request.form.get('use_verification', 'false').lower() == 'true'

    if use_verification:
        return verify()

    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]

    prob = model.predict_proba(vector)[0]
    confidence = round(max(prob) * 100, 2)

    result = "Fake News" if prediction == 0 else "Real News"

    return render_template("index.html", prediction=result, confidence=confidence)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# ---------------- PWA ASSETS ----------------
@app.route('/manifest.json')
def manifest():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'manifest.json', as_attachment=False)

@app.route('/sw.js')
def service_worker():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sw.js', as_attachment=False)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'icon-192.png', as_attachment=False)

# ---------------- RUN ----------------
init_db()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000 ,debug=True)