Here's a clean README for the project:

---

# ConvoLens 🔍

**AI-Powered Behavioral & Emotional Pattern Analyzer for Text Conversations**

> Analyzes WhatsApp-style chats using NLP to detect behavioral patterns like defensiveness, manipulation, avoidance, and emotional inconsistency. Does **not** claim to detect lying or cheating — only conversational patterns.

---

## Setup

**Requirements:** Python 3.10+, VS Code

```bash
# 1. Open the convolens folder in VS Code

# 2. Create virtual environment
python -m venv venv

# 3. Activate it (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## How to Use

1. Choose an input method from the sidebar:
   - Upload a WhatsApp `.txt` export
   - Paste conversation manually (`Person: message` format)
   - Load a built-in sample chat
2. Click **Analyze**
3. Explore the 5 tabs — Dashboard, Sentiment, Patterns, Visualizations, Report

---

## What It Detects

| Pattern | Description |
|---|---|
| Defensiveness | Blame-shifting, denial language |
| Manipulation | Guilt-tripping, gaslighting phrases |
| Avoidance | One-word replies, dismissive language |
| Toxicity | Harsh or aggressive language |
| Emotional inconsistency | Sudden mood swings in messages |
| Low engagement | Short, disinterested responses |

---

## Project Structure

```
convolens/
├── app.py                 ← Run this
├── requirements.txt
├── core/
│   ├── parser.py          ← Chat parsing
│   ├── preprocessor.py    ← Text cleaning
│   ├── sentiment_analyzer.py
│   ├── pattern_detector.py
│   ├── tone_analyzer.py
│   ├── scorer.py
│   └── report_generator.py
├── data/
│   ├── keywords.py        ← Behavioral keyword lists
│   └── sample_chats/
└── ui/
    └── charts.py          ← All visualizations
```

---

## Tech Stack

- **Streamlit** — UI framework
- **VADER**(Valence Aware Dictionary and sEntiment Reasoner) — Sentiment analysis
- **NLTK** — Tokenization, lemmatization
- **scikit-learn** — TF-IDF, cosine similarity
- **Plotly** — Interactive charts
- **Pandas** — Data handling

---

## Ethical Notice

This tool identifies linguistic and behavioral patterns in text using NLP. It does not scientifically determine intent, truthfulness, or relationship quality. All results are observational, not conclusive.

---

*College AI/ML Project — NLP Track*
