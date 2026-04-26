# 🕌 Islamic Scholar Recommendation System
## Zero se Deployment tak — Complete Guide

---

## 📦 Files ka Structure

```
islamic_recommender/
├── app.py                        ← Main Flask app (backend + frontend)
├── islamic_scholars_dataset.csv  ← 213 rows ka dataset
├── requirements.txt              ← Python libraries
├── Procfile                      ← Deployment ke liye
└── README.md                     ← Yeh file
```

---

## 🔰 STEP 1: Python Install Karo

1. https://python.org/downloads pe jao
2. Python 3.10+ download karo
3. Install karte waqt **"Add to PATH"** checkbox zaroor tick karo
4. Check karo: terminal mein likho → `python --version`

---

## 🔰 STEP 2: Project Folder Banao

```bash
# Koi bhi jagah folder banao
mkdir islamic_recommender
cd islamic_recommender

# Yahan saari files rakho:
# app.py
# islamic_scholars_dataset.csv
# requirements.txt
# Procfile
```

---

## 🔰 STEP 3: Virtual Environment (Recommended)

```bash
# Virtual env banao
python -m venv venv

# Windows pe activate karo:
venv\Scripts\activate

# Mac/Linux pe activate karo:
source venv/bin/activate

# Aapko (venv) dikhega terminal mein — matlab active hai
```

---

## 🔰 STEP 4: Libraries Install Karo

```bash
pip install -r requirements.txt

# Ya manually:
pip install flask pandas numpy scikit-learn gunicorn
```

---

## 🔰 STEP 5: Local Run Karo

```bash
python app.py
```

Output dikhega:
```
============================================================
  🕌 Islamic Scholar Recommender
  📊 Dataset: 213 rows loaded
  🔢 TF-IDF vocab: XXXX terms
  🌐 Open: http://localhost:5000
============================================================
```

Browser mein kholo: **http://localhost:5000**

---

## 🔰 STEP 6: Use Karo

1. Search box mein apna sawal likho
2. "Talash کریں" button dabao
3. Top 3 scholars recommend honge with:
   - Match % score
   - Scholar style & audience
   - Best matching answer
   - YouTube channel
   - Topics covered

### Example Queries:
- `Best scholar for emotional bayans`
- `Quran tafsir Urdu mein`
- `Comparative religion debate`
- `طارق جمیل کی تقریر`
- `Youth aur modern challenges Islam`
- `Islamic law modern fiqh`

---

## 🚀 DEPLOYMENT — Render.com (Free)

### Step 1: GitHub pe Upload Karo

1. https://github.com pe account banao
2. New repository banao: `islamic-scholar-recommender`
3. Saari files upload karo (app.py, requirements.txt, Procfile, CSV)

### Step 2: Render pe Deploy Karo

1. https://render.com pe jao → Sign up with GitHub
2. "New" → "Web Service" click karo
3. Apna GitHub repo select karo
4. Settings:
   ```
   Name:         islamic-scholar-recommender
   Environment:  Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
   ```
5. "Create Web Service" click karo
6. 2-3 minute wait karo
7. URL milegi: `https://islamic-scholar-recommender.onrender.com`

---

## 🚀 DEPLOYMENT — Railway.app (Alternative)

```bash
# Railway CLI install karo
npm install -g @railway/cli

# Login karo
railway login

# Project banao
railway init

# Deploy karo
railway up
```

---

## 🧠 System Kaise Kaam Karta Hai

```
User Query
    ↓
TF-IDF Vectorizer (query ko numbers mein convert karta hai)
    ↓
Cosine Similarity (query ko dataset ke har row se compare karta hai)
    ↓
Scholar ke scores aggregate hote hain
    ↓
Top 3 scholars ranked by average similarity score
    ↓
Result with confidence % show karta hai
```

### Algorithm:
- **TF-IDF**: Term Frequency-Inverse Document Frequency
  - Common words ko kam importance deta hai
  - Rare/specific words ko zyada importance deta hai
- **Cosine Similarity**: Do vectors ke beech ka angle measure karta hai
  - 1 = perfect match, 0 = koi match nahi

---

## 📊 Dataset Info

| Column | Description |
|--------|-------------|
| query | User ka sawal |
| scholar_name | Scholar ka naam (English) |
| scholar_urdu | Scholar ka naam (Urdu) |
| category | biography/lectures/books/recommendation/comparison/topic |
| topic | Specific topic |
| intent | information/recommendation/comparison/lecture_search |
| language | urdu/english/mixed |
| answer | Detailed jawab |
| label | Target class (scholar name) |
| label_id | Numeric label 0-5 |

---

## 🔧 API Endpoints

```
GET  /              → Web UI
POST /recommend     → Scholar recommendations
GET  /dataset/stats → Dataset statistics
```

### POST /recommend example:
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "best tafsir scholar", "top_n": 3}'
```

Response:
```json
{
  "query": "best tafsir scholar",
  "recommendations": [
    {
      "scholar_name": "Israr Ahmed",
      "scholar_urdu": "اسرار احمد",
      "confidence": 23.5,
      "style": "Academic & Revolutionary",
      "audience": "Educated Muslims, Youth",
      ...
    }
  ]
}
```

---

## 🔄 Aage Improve Karne ke Liye

1. **BERT Fine-tuning**: `Islamic_Scholars_BERT_Finetune.ipynb` use karo
2. **More Data**: `generate_islamic_dataset.py` se 500+ rows generate karo
3. **User Feedback**: Thumbs up/down system add karo
4. **More Scholars**: `SCHOLARS_INFO` dict mein add karo
5. **Database**: SQLite ya PostgreSQL add karo

---

## ❓ Mushkilat?

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` dobara chalao |
| Port already in use | `python app.py` mein port change karo: `app.run(port=5001)` |
| CSV not found | CSV file app.py ke saath same folder mein rakho |
| Slow on first request | Normal hai — TF-IDF pehli baar load hota hai |
