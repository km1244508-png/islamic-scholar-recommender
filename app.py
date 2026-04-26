"""
Islamic Scholar Recommendation System
Full Stack: Flask backend + TF-IDF + Cosine Similarity
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import json, os, re

app = Flask(__name__)

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "islamic_scholars_dataset.csv")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

SCHOLARS_INFO = {
    "Tariq Jameel": {
        "urdu": "طارق جمیل",
        "born": 1953, "country": "Pakistan",
        "style": "Emotional & Spiritual",
        "audience": "General Muslim Public",
        "specialization": ["Tablighi Dawah", "Spiritual Reform", "Hadith"],
        "known_for": "Emotional bayans, celebrity conversions, Tablighi Jamaat",
        "youtube": "Maulana Tariq Jameel Official",
        "followers": "8.5M+",
        "languages": ["Urdu"],
        "topics": ["Tawbah","Akhirat","Seerat","Islah","Duniya","Rishtay"],
        "emoji": "🌙",
        "color": "#1a4a2e",
        "accent": "#4ade80",
        "image_text": "TJ"
    },
    "Ibtasim Illahi Zaheer": {
        "urdu": "ابتسام الٰہی ظہیر",
        "born": 1975, "country": "Pakistan",
        "style": "Scholarly & Analytical",
        "audience": "Religious Students, Scholars",
        "specialization": ["Ahl-e-Hadith Fiqh", "Comparative Religion", "Dawah"],
        "known_for": "Scholarly debates, Ahl-e-Hadith scholarship",
        "youtube": "Ibtasim Ilahi Zaheer Official",
        "followers": "2.1M+",
        "languages": ["Urdu", "Arabic", "English"],
        "topics": ["Fiqh","Aqeedah","Hadith Science","Comparative Religion","Family Law"],
        "emoji": "📚",
        "color": "#1a2a4a",
        "accent": "#60a5fa",
        "image_text": "IZ"
    },
    "Israr Ahmed": {
        "urdu": "اسرار احمد",
        "born": 1932, "country": "Pakistan",
        "style": "Academic & Revolutionary",
        "audience": "Educated Muslims, Youth",
        "specialization": ["Quran Tafsir", "Islamic Revival", "Khilafah"],
        "known_for": "Tanzeem-e-Islami, detailed Quranic tafsir",
        "youtube": "Dr Israr Ahmed Official",
        "followers": "3.2M+",
        "languages": ["Urdu"],
        "topics": ["Tafsir","Islamic State","Quran","Iman","Jihad bil Nafs"],
        "emoji": "📖",
        "color": "#2a1a0a",
        "accent": "#f59e0b",
        "image_text": "IA"
    },
    "Zakir Naik": {
        "urdu": "ذاکر نائک",
        "born": 1965, "country": "India",
        "style": "Logical & Debate",
        "audience": "Non-Muslims, English Speaking",
        "specialization": ["Comparative Religion", "Dawah", "Science and Islam"],
        "known_for": "Peace TV, IRF, memorizing religious scriptures",
        "youtube": "Dr Zakir Naik",
        "followers": "22M+",
        "languages": ["English", "Urdu", "Arabic"],
        "topics": ["Comparative Religion","Science Islam","Bible Quran","Women in Islam"],
        "emoji": "⚖️",
        "color": "#1a1a2a",
        "accent": "#a78bfa",
        "image_text": "ZN"
    },
    "Javed Ahmad Ghamidi": {
        "urdu": "جاوید احمد غامدی",
        "born": 1951, "country": "Pakistan",
        "style": "Intellectual & Reform",
        "audience": "Educated Elite, Doubting Muslims",
        "specialization": ["Quran Tafsir", "Islamic Law Reform", "Modern Fiqh"],
        "known_for": "Al-Mawrid Institute, Mizan book",
        "youtube": "Ghamidi TV",
        "followers": "1.8M+",
        "languages": ["Urdu", "English"],
        "topics": ["Modern Fiqh","Quran Interpretation","Women Rights","Democracy Islam"],
        "emoji": "🔬",
        "color": "#0a1a2a",
        "accent": "#22d3ee",
        "image_text": "JG"
    },
    "Nouman Ali Khan": {
        "urdu": "نعمان علی خان",
        "born": 1978, "country": "USA",
        "style": "Relatable & Youth Oriented",
        "audience": "Western Muslims, Youth, Students",
        "specialization": ["Quran Arabic", "Youth Dawah", "Tafsir"],
        "known_for": "Bayyinah Institute, Arabic grammar, youth connection",
        "youtube": "Nouman Ali Khan Official",
        "followers": "5.5M+",
        "languages": ["English", "Urdu"],
        "topics": ["Quran Arabic","Youth Issues","Marriage","Faith Crisis","Modern Challenges"],
        "emoji": "✨",
        "color": "#1a0a2a",
        "accent": "#f472b6",
        "image_text": "NK"
    },
}

# ─── TF-IDF MODEL ──────────────────────────────────────────────────────────────
df["combined_text"] = (
    df["query"].fillna("") + " " +
    df["answer"].fillna("") + " " +
    df["scholar_name"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["topic"].fillna("")
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words=None,
    min_df=1
)
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["scholar_name"])


def recommend(query: str, top_n: int = 3) -> list[dict]:
    """Given a query, return top_n scholar recommendations."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Aggregate scores per scholar
    scholar_scores = {}
    for idx, score in enumerate(scores):
        scholar = df.iloc[idx]["scholar_name"]
        if scholar not in scholar_scores:
            scholar_scores[scholar] = {"total": 0.0, "count": 0, "best_answer": "", "best_score": 0}
        scholar_scores[scholar]["total"] += score
        scholar_scores[scholar]["count"] += 1
        if score > scholar_scores[scholar]["best_score"]:
            scholar_scores[scholar]["best_score"] = score
            scholar_scores[scholar]["best_answer"] = df.iloc[idx]["answer"]

    # Sort by average score
    ranked = sorted(
        scholar_scores.items(),
        key=lambda x: x[1]["total"] / max(x[1]["count"], 1),
        reverse=True
    )[:top_n]

    results = []
    for name, data in ranked:
        info = SCHOLARS_INFO.get(name, {})
        avg_score = data["total"] / max(data["count"], 1)
        results.append({
            "scholar_name": name,
            "scholar_urdu": info.get("urdu", ""),
            "confidence": round(float(avg_score) * 100, 1),
            "best_match_score": round(float(data["best_score"]) * 100, 1),
            "answer": data["best_answer"],
            "style": info.get("style", ""),
            "audience": info.get("audience", ""),
            "specialization": info.get("specialization", []),
            "known_for": info.get("known_for", ""),
            "youtube": info.get("youtube", ""),
            "followers": info.get("followers", ""),
            "languages": info.get("languages", []),
            "topics": info.get("topics", []),
            "emoji": info.get("emoji", "🌙"),
            "color": info.get("color", "#1a1a1a"),
            "accent": info.get("accent", "#4ade80"),
            "image_text": info.get("image_text", "??"),
            "born": info.get("born", ""),
            "country": info.get("country", ""),
        })
    return results


# ─── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Islamic Scholar Recommender — علماء اسلام</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Amiri:wght@400;700&family=Source+Serif+4:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #0a0a0f;
  --surface: #111118;
  --border: #1e1e2e;
  --text: #e8e8f0;
  --muted: #6b6b8a;
  --gold: #c9a84c;
  --gold-light: #f0d080;
  --green: #2d6a4f;
  --radius: 16px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Source Serif 4', serif;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── BACKGROUND PATTERN ── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 20%, rgba(201,168,76,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 60% at 80% 80%, rgba(45,106,79,0.06) 0%, transparent 60%);
  pointer-events: none;
}

/* ── GEOMETRIC ORNAMENT ── */
.ornament {
  position: fixed; top: 0; right: 0; width: 400px; height: 400px;
  opacity: 0.04; z-index: 0; pointer-events: none;
  background-image: repeating-linear-gradient(
    60deg,
    var(--gold) 0, var(--gold) 1px,
    transparent 0, transparent 50%
  );
  background-size: 30px 30px;
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
}

main { position: relative; z-index: 1; }

/* ── HEADER ── */
header {
  text-align: center;
  padding: 64px 24px 40px;
  border-bottom: 1px solid var(--border);
  position: relative;
}
.header-bismillah {
  font-family: 'Amiri', serif;
  font-size: clamp(28px, 5vw, 48px);
  color: var(--gold);
  letter-spacing: 0.05em;
  margin-bottom: 12px;
  text-shadow: 0 0 40px rgba(201,168,76,0.3);
}
.header-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(28px, 5vw, 52px);
  font-weight: 900;
  line-height: 1.1;
  background: linear-gradient(135deg, var(--gold-light) 0%, var(--gold) 50%, #a07830 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}
.header-sub {
  font-family: 'Amiri', serif;
  font-size: clamp(16px, 2.5vw, 22px);
  color: var(--muted);
  letter-spacing: 0.1em;
}
.header-divider {
  width: 120px; height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  margin: 20px auto 0;
}

/* ── SEARCH SECTION ── */
.search-wrap {
  max-width: 780px;
  margin: 48px auto 0;
  padding: 0 24px;
}
.search-label {
  display: block;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: var(--gold);
  margin-bottom: 12px;
}
.search-box {
  display: flex; gap: 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 6px 6px 6px 20px;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.search-box:focus-within {
  border-color: var(--gold);
  box-shadow: 0 0 0 3px rgba(201,168,76,0.15), 0 8px 32px rgba(0,0,0,0.4);
}
.search-input {
  flex: 1;
  background: none; border: none; outline: none;
  color: var(--text);
  font-family: 'Source Serif 4', serif;
  font-size: 17px;
  padding: 10px 0;
}
.search-input::placeholder { color: var(--muted); }
.search-btn {
  background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%);
  color: #0a0a0f;
  border: none; cursor: pointer;
  font-family: 'Source Serif 4', serif;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 12px 24px;
  border-radius: 10px;
  transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
  white-space: nowrap;
}
.search-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(201,168,76,0.4); }
.search-btn:active { transform: translateY(0); }
.search-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

/* ── EXAMPLE PILLS ── */
.examples {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-top: 16px;
}
.pill {
  background: rgba(201,168,76,0.08);
  border: 1px solid rgba(201,168,76,0.2);
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 13px;
  color: var(--gold);
  cursor: pointer;
  transition: all 0.2s;
  font-family: 'Source Serif 4', serif;
}
.pill:hover { background: rgba(201,168,76,0.16); border-color: var(--gold); }

/* ── RESULTS ── */
.results-wrap {
  max-width: 1100px;
  margin: 48px auto 80px;
  padding: 0 24px;
}
.results-title {
  font-family: 'Playfair Display', serif;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.25em;
  color: var(--muted);
  margin-bottom: 24px;
}
.results-title span { color: var(--gold); }
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

/* ── SCHOLAR CARD ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
  animation: fadeUp 0.5s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
.card:nth-child(1) { animation-delay: 0.05s; }
.card:nth-child(2) { animation-delay: 0.12s; }
.card:nth-child(3) { animation-delay: 0.19s; }
.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.card-header {
  padding: 28px 24px 20px;
  display: flex; align-items: center; gap: 16px;
  position: relative;
}
.card-header::after {
  content: '';
  position: absolute; bottom: 0; left: 24px; right: 24px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.card-avatar {
  width: 56px; height: 56px;
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Playfair Display', serif;
  font-size: 20px;
  font-weight: 900;
  flex-shrink: 0;
  position: relative;
}
.rank-badge {
  position: absolute; top: -6px; right: -6px;
  width: 20px; height: 20px;
  background: var(--gold);
  color: #0a0a0f;
  border-radius: 50%;
  font-size: 10px;
  font-weight: 700;
  font-family: 'Source Serif 4', serif;
  display: flex; align-items: center; justify-content: center;
}
.card-names { flex: 1; min-width: 0; }
.card-name {
  font-family: 'Playfair Display', serif;
  font-size: 20px;
  font-weight: 700;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.card-urdu {
  font-family: 'Amiri', serif;
  font-size: 15px;
  color: var(--muted);
  margin-top: 2px;
  direction: rtl;
}
.confidence-wrap {
  text-align: right; flex-shrink: 0;
}
.confidence-num {
  font-family: 'Playfair Display', serif;
  font-size: 26px;
  font-weight: 900;
  line-height: 1;
}
.confidence-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted); }

.card-body { padding: 20px 24px; }
.meta-row {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-bottom: 16px;
}
.meta-chip {
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 6px;
  font-family: 'Source Serif 4', serif;
  letter-spacing: 0.05em;
}
.answer-text {
  font-size: 14px;
  line-height: 1.7;
  color: #b0b0c8;
  margin-bottom: 16px;
  border-left: 2px solid var(--border);
  padding-left: 12px;
}
.tags-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.tag {
  font-size: 11px;
  padding: 3px 8px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--muted);
}
.card-footer {
  padding: 12px 24px 16px;
  border-top: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px;
}
.yt-link {
  font-size: 12px;
  color: var(--muted);
  text-decoration: none;
  display: flex; align-items: center; gap: 6px;
  transition: color 0.2s;
}
.yt-link:hover { color: #ff4444; }
.followers-badge {
  font-size: 12px;
  color: var(--gold);
  background: rgba(201,168,76,0.1);
  padding: 4px 10px;
  border-radius: 999px;
}

/* ── LOADING ── */
.loading {
  text-align: center; padding: 60px;
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 0.5 } 50% { opacity: 1 } }
.loading-arabic {
  font-family: 'Amiri', serif;
  font-size: 32px;
  color: var(--gold);
}
.loading-text { font-size: 14px; color: var(--muted); margin-top: 12px; }

/* ── EMPTY STATE ── */
.empty {
  text-align: center; padding: 80px 24px;
  color: var(--muted);
}
.empty-icon { font-size: 48px; margin-bottom: 16px; }
.empty-text { font-size: 16px; }

/* ── STATS BAR ── */
.stats-bar {
  display: flex; justify-content: center; gap: 40px;
  padding: 20px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.stat { text-align: center; }
.stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 24px;
  font-weight: 700;
  color: var(--gold);
}
.stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted); }

/* ── RESPONSIVE ── */
@media (max-width: 600px) {
  .search-box { flex-direction: column; padding: 12px; }
  .search-btn { width: 100%; text-align: center; }
  .cards-grid { grid-template-columns: 1fr; }
  .stats-bar { gap: 24px; }
}
</style>
</head>
<body>
<div class="ornament"></div>
<main>

<header>
  <div class="header-bismillah">بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ</div>
  <h1 class="header-title">Islamic Scholar Recommender</h1>
  <p class="header-sub">علماء اسلام — آپ کے لیے بہترین عالم تلاش کریں</p>
  <div class="header-divider"></div>
</header>

<div class="stats-bar">
  <div class="stat"><div class="stat-num">6</div><div class="stat-label">Scholars</div></div>
  <div class="stat"><div class="stat-num">213</div><div class="stat-label">Dataset Rows</div></div>
  <div class="stat"><div class="stat-num">3</div><div class="stat-label">Languages</div></div>
  <div class="stat"><div class="stat-num">TF-IDF</div><div class="stat-label">Algorithm</div></div>
</div>

<div class="search-wrap">
  <label class="search-label" for="qinput">🔍 Apna sawal likhein — Urdu, English, ya Arabic mein</label>
  <div class="search-box">
    <input
      id="qinput"
      class="search-input"
      type="text"
      placeholder="e.g. Quran tafsir ke liye best scholar kaun hai?"
      autocomplete="off"
    />
    <button class="search-btn" onclick="search()" id="sbtn">Talash کریں</button>
  </div>
  <div class="examples">
    <span class="pill" onclick="setQuery('Best scholar for emotional bayans')">Emotional bayans</span>
    <span class="pill" onclick="setQuery('Comparative religion debate')">Comparative Religion</span>
    <span class="pill" onclick="setQuery('Quran tafsir Urdu mein')">Quran Tafsir</span>
    <span class="pill" onclick="setQuery('Youth aur modern challenges')">Youth Issues</span>
    <span class="pill" onclick="setQuery('Islamic law reform fiqh')">Modern Fiqh</span>
    <span class="pill" onclick="setQuery('طارق جمیل کی تقریر')">طارق جمیل</span>
    <span class="pill" onclick="setQuery('Hadith science scholarly')">Hadith Science</span>
  </div>
</div>

<div class="results-wrap" id="results"></div>

</main>

<script>
const COLORS = {
  "Tariq Jameel":         { bg:"#1a4a2e", accent:"#4ade80" },
  "Ibtasim Illahi Zaheer":{ bg:"#1a2a4a", accent:"#60a5fa" },
  "Israr Ahmed":          { bg:"#2a1a0a", accent:"#f59e0b" },
  "Zakir Naik":           { bg:"#1a1a2a", accent:"#a78bfa" },
  "Javed Ahmad Ghamidi":  { bg:"#0a1a2a", accent:"#22d3ee" },
  "Nouman Ali Khan":      { bg:"#1a0a2a", accent:"#f472b6" },
};

function setQuery(q) {
  document.getElementById('qinput').value = q;
  search();
}

document.getElementById('qinput').addEventListener('keydown', e => {
  if (e.key === 'Enter') search();
});

async function search() {
  const q = document.getElementById('qinput').value.trim();
  if (!q) return;

  const btn = document.getElementById('sbtn');
  btn.disabled = true; btn.textContent = 'تلاش...';

  const div = document.getElementById('results');
  div.innerHTML = `<div class="loading">
    <div class="loading-arabic">☽</div>
    <div class="loading-text">Scholars ko talash kar rahe hain...</div>
  </div>`;

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query: q, top_n: 3})
    });
    const data = await res.json();

    if (!data.recommendations || data.recommendations.length === 0) {
      div.innerHTML = `<div class="empty">
        <div class="empty-icon">🔍</div>
        <div class="empty-text">Koi result nahi mila. Dusra sawal likhein.</div>
      </div>`;
      return;
    }

    const c = COLORS;
    div.innerHTML = `<div class="results-title">
      Aapki query ke liye <span>${data.recommendations.length} best scholars</span> mile:
    </div>
    <div class="cards-grid">
      ${data.recommendations.map((s, i) => renderCard(s, i+1)).join('')}
    </div>`;

  } catch(e) {
    div.innerHTML = `<div class="empty">
      <div class="empty-icon">⚠️</div>
      <div class="empty-text">Error: ${e.message}</div>
    </div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Talash کریں';
  }
}

function renderCard(s, rank) {
  const col = COLORS[s.scholar_name] || {bg:"#1a1a1a", accent:"#c9a84c"};
  const confColor = s.confidence > 15 ? col.accent : '#6b6b8a';
  const specs = (s.specialization||[]).slice(0,2).map(sp =>
    `<span class="meta-chip" style="background:${col.bg};color:${col.accent};border:1px solid ${col.accent}33">${sp}</span>`
  ).join('');
  const topics = (s.topics||[]).slice(0,4).map(t =>
    `<span class="tag">${t}</span>`
  ).join('');
  const langs = (s.languages||[]).map(l =>
    `<span class="meta-chip" style="background:rgba(255,255,255,0.04);color:#888;border:1px solid #333">${l}</span>`
  ).join('');

  return `
  <div class="card" style="border-color:${col.accent}22">
    <div class="card-header">
      <div class="card-avatar" style="background:${col.bg};color:${col.accent}">
        ${s.emoji || s.image_text}
        <div class="rank-badge">${rank}</div>
      </div>
      <div class="card-names">
        <div class="card-name">${s.scholar_name}</div>
        <div class="card-urdu">${s.scholar_urdu}</div>
        <div style="font-size:11px;color:#555;margin-top:4px">${s.country} · ${s.born}</div>
      </div>
      <div class="confidence-wrap">
        <div class="confidence-num" style="color:${confColor}">${s.confidence}%</div>
        <div class="confidence-label">Match</div>
      </div>
    </div>
    <div class="card-body">
      <div class="meta-row">${specs}${langs}</div>
      <div style="font-size:12px;color:#666;margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em">Style: <span style="color:${col.accent}">${s.style}</span></div>
      <div style="font-size:12px;color:#666;margin-bottom:14px">Audience: ${s.audience}</div>
      ${s.answer ? `<div class="answer-text">${s.answer.slice(0,180)}${s.answer.length>180?'...':''}</div>` : ''}
      <div class="tags-row">${topics}</div>
    </div>
    <div class="card-footer">
      <span class="yt-link">▶ ${s.youtube}</span>
      <span class="followers-badge">👥 ${s.followers}</span>
    </div>
  </div>`;
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_n = int(data.get("top_n", 3))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = recommend(query, top_n=top_n)
    return jsonify({"query": query, "recommendations": results})


@app.route("/dataset/stats")
def dataset_stats():
    return jsonify({
        "total_rows": len(df),
        "scholars": df["scholar_name"].value_counts().to_dict(),
        "categories": df["category"].value_counts().to_dict(),
        "languages": df["language"].value_counts().to_dict(),
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  🕌 Islamic Scholar Recommender")
    print(f"  📊 Dataset: {len(df)} rows loaded")
    print(f"  🔢 TF-IDF vocab: {len(vectorizer.vocabulary_)} terms")
    print("  🌐 Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
