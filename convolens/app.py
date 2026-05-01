import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.parser import parse_whatsapp_chat, parse_manual_input, get_speakers
from core.preprocessor import preprocess_dataframe
from core.sentiment_analyzer import analyze_sentiment_dataframe, get_speaker_sentiment_summary
from core.pattern_detector import analyze_patterns_dataframe, get_speaker_pattern_summary
from core.tone_analyzer import detect_tone_shifts, compute_tfidf_keywords
from core.scorer import compute_risk_score, compute_conversation_health
from core.report_generator import generate_report
from ui.charts import (sentiment_timeline_chart, behavioral_radar_chart,
                        speaker_contribution_pie, toxicity_trend_chart,
                        emotional_volatility_chart, risk_gauge)

st.set_page_config(
    page_title="ConvoLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #5C6BC0; margin-bottom: 0;}
    .sub-header {color: #888; font-size: 1rem; margin-bottom: 2rem;}
    .metric-card {background: #f8f9ff; border-radius: 12px; padding: 1.2rem;
                   border-left: 4px solid #5C6BC0; margin-bottom: 1rem;}
    .ethical-note {background: #e8f4fd; border-radius: 8px; padding: 1rem;
                    border-left: 3px solid #2196F3; font-size: 0.85rem; color: #555;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔍 ConvoLens</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Behavioral & Emotional Pattern Analyzer</p>', unsafe_allow_html=True)

st.markdown("""
<div class="ethical-note">
⚠️ <b>Ethical Notice:</b> ConvoLens identifies behavioral and linguistic patterns using NLP techniques only.
It does <b>not</b> scientifically determine intent, truthfulness, or relationship quality.
Use results as observational insights, not conclusions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Input Configuration")
    input_mode = st.radio("Choose input method:",
                           ["Upload WhatsApp .txt", "Manual text input", "Load sample chat"])
    st.markdown("---")
    st.markdown("**Analysis Settings**")
    tone_window = st.slider("Tone shift window size", 3, 10, 5,
                              help="Number of messages to compare for tone shift detection")
    st.markdown("---")
    st.markdown("**About ConvoLens**")
    st.markdown("Built with Python, NLTK, VADER, scikit-learn & Streamlit")
    st.markdown("*College AI/ML Project — NLP Track*")

df = None

if input_mode == "Upload WhatsApp .txt":
    uploaded = st.file_uploader("Upload WhatsApp chat export (.txt)", type=['txt'])
    if uploaded:
        text = uploaded.read().decode('utf-8', errors='ignore')
        df = parse_whatsapp_chat(text)
        if df.empty:
            st.error("Could not parse chat. Ensure it is a standard WhatsApp export format.")

elif input_mode == "Manual text input":
    st.markdown("**Format:** Each line should be `PersonName: message text`")
    sample = "Alice: Hey are you coming tonight?\nBob: whatever, i don't care\nAlice: You always say that\nBob: Stop blaming me for everything\nAlice: I'm just asking\nBob: Fine. Sure. I'll come if you want me to"
    manual_text = st.text_area("Paste conversation:", value=sample, height=200)
    if st.button("Analyze conversation", type="primary"):
        df = parse_manual_input(manual_text)

elif input_mode == "Load sample chat":
    sample_choice = st.selectbox("Choose sample:", ["Tense relationship chat", "Healthy friendly chat"])
    if st.button("Analyze this sample", type="primary"):
        base = os.path.dirname(os.path.abspath(__file__))
        if sample_choice == "Tense relationship chat":
            path = os.path.join(base, "data", "sample_chats", "sample_tense.txt")
        else:
            path = os.path.join(base, "data", "sample_chats", "sample_healthy.txt")
        with open(path, 'r') as f:
            df = parse_manual_input(f.read())

if df is not None and not df.empty:
    with st.spinner("Running NLP analysis pipeline..."):
        df = preprocess_dataframe(df)
        df = analyze_sentiment_dataframe(df)
        df = analyze_patterns_dataframe(df)
        df = detect_tone_shifts(df, window_size=tone_window)

        speakers = get_speakers(df)
        speaker_sentiment = get_speaker_sentiment_summary(df)
        speaker_patterns = get_speaker_pattern_summary(df)
        risk_scores = {s: compute_risk_score(df, s) for s in speakers}
        health = compute_conversation_health(df)
        keywords = compute_tfidf_keywords(df)
        report = generate_report(df, speaker_sentiment, speaker_patterns, risk_scores, health)

    st.success(f"Analysis complete — {len(df)} messages processed")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(df))
    with col2:
        st.metric("Participants", len(speakers))
    with col3:
        st.metric("Conversation Health", f"{health['score']}/100", delta=health['label'])
    with col4:
        avg_risk = sum(r['total'] for r in risk_scores.values()) / max(len(risk_scores), 1)
        st.metric("Avg Risk Score", f"{avg_risk:.1f}/100")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "💬 Sentiment", "🧠 Behavioral Patterns",
        "📈 Visualizations", "📄 Full Report"
    ])

    with tab1:
        st.subheader("Risk Score Overview")
        gauge_cols = st.columns(min(len(speakers), 2))
        for i, speaker in enumerate(speakers[:2]):
            with gauge_cols[i]:
                st.plotly_chart(risk_gauge(risk_scores[speaker]['total'], speaker),
                                use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(speaker_contribution_pie(df), use_container_width=True)
        with col_b:
            for speaker in speakers[:2]:
                r = risk_scores[speaker]
                st.markdown(f"""
                <div class="metric-card">
                    <b>{speaker}</b><br>
                    Risk: <b>{r['total']}/100</b> — {r['label']}<br>
                    <small>Sentiment: {r['sentiment_component']} |
                    Patterns: {r['pattern_component']} |
                    Tone shifts: {r['tone_component']}</small>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader("Sentiment Analysis")
        st.plotly_chart(sentiment_timeline_chart(df), use_container_width=True)
        st.subheader("Per-Speaker Sentiment Breakdown")
        sent_cols = st.columns(min(len(speakers), 2))
        for i, speaker in enumerate(speakers[:2]):
            with sent_cols[i]:
                s = speaker_sentiment[speaker]
                st.markdown(f"**{speaker}**")
                st.metric("Avg Sentiment", f"{s['avg_compound']:+.2f}")
                total = s['total_msgs']
                st.progress(s['positive_msgs'] / total, text=f"Positive: {s['positive_msgs']}")
                st.progress(s['negative_msgs'] / total, text=f"Negative: {s['negative_msgs']}")
                st.progress(s['neutral_msgs'] / total, text=f"Neutral: {s['neutral_msgs']}")

    with tab3:
        st.subheader("Behavioral Pattern Analysis")
        st.plotly_chart(behavioral_radar_chart(speaker_patterns), use_container_width=True)
        for speaker in speakers[:2]:
            with st.expander(f"🔍 {speaker} — Detailed Pattern Breakdown"):
                patterns = speaker_patterns[speaker]
                pattern_df = pd.DataFrame([
                    {
                        'Pattern': k.replace('_', ' ').title(),
                        'Occurrences': v['occurrence_count'],
                        'Frequency (%)': v['frequency_pct'],
                        'Total Score': v['total_score']
                    }
                    for k, v in patterns.items()
                ]).sort_values('Total Score', ascending=False)
                st.dataframe(pattern_df, use_container_width=True, hide_index=True)

        st.subheader("Flagged Messages")
        flagged = df[df['pattern_total_score'] > 2].copy()
        if not flagged.empty:
            for _, row in flagged.head(10).iterrows():
                patterns_str = ', '.join(row['patterns_detected']) if row['patterns_detected'] else 'none'
                st.markdown(f"""
                <div style="background:#fff5f5;border-left:3px solid #EF5350;
                padding:0.7rem;border-radius:6px;margin-bottom:0.5rem;font-size:0.9rem;">
                    <b>{row['speaker']}</b>: {row['message']}<br>
                    <small style="color:#888">Patterns: {patterns_str} | Score: {row['pattern_total_score']:.1f}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No strongly flagged messages found.")

    with tab4:
        st.plotly_chart(toxicity_trend_chart(df), use_container_width=True)
        st.plotly_chart(emotional_volatility_chart(df), use_container_width=True)
        st.subheader("Tone Shift Points")
        shifts = df[df['is_tone_shift'] == True][['speaker', 'message', 'tone_shift_score']].head(10)
        if not shifts.empty:
            st.dataframe(shifts, use_container_width=True, hide_index=True)
        else:
            st.info("No significant tone shifts detected.")

    with tab5:
        st.markdown(report)
        st.download_button(
            label="Download Report (.md)",
            data=report,
            file_name="convolens_report.md",
            mime="text/markdown"
        )

else:
    st.markdown("""
    ### How to use ConvoLens
    1. **Export** your WhatsApp chat: Chat → ⋮ → More → Export chat (without media)
    2. **Upload** the .txt file, or paste conversation text manually
    3. **Get** your behavioral analysis report instantly

    **What gets analyzed:**
    - Sentiment polarity per message (VADER)
    - Defensiveness, manipulation, and avoidance patterns
    - Emotional volatility and tone shifts (cosine similarity)
    - Toxicity signals
    - Conversation health score (composite)
    """)
