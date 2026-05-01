import pandas as pd


def compute_risk_score(df: pd.DataFrame, speaker: str) -> dict:
    speaker_df = df[df['speaker'] == speaker]
    if speaker_df.empty:
        return {'total': 0}

    total_msgs = len(speaker_df)

    avg_compound = speaker_df['sentiment_compound'].mean()
    negativity_raw = max(0, -avg_compound)
    volatility = speaker_df['sentiment_compound'].std()
    sentiment_score = min(25, (negativity_raw * 15) + (volatility * 10))

    avg_pattern_score = speaker_df['pattern_total_score'].mean()
    pattern_score = min(40, avg_pattern_score * 8)

    if 'tone_shift_score' in speaker_df.columns:
        tone_shift_rate = (speaker_df['is_tone_shift'] == True).sum() / total_msgs
        tone_score = min(20, tone_shift_rate * 100)
    else:
        tone_score = 0

    avg_word_count = speaker_df['word_count'].mean()
    low_engagement_score = min(15, max(0, (5 - avg_word_count) * 3)) if avg_word_count < 5 else 0

    total_score = sentiment_score + pattern_score + tone_score + low_engagement_score
    total_score = round(min(100, max(0, total_score)), 1)

    if total_score < 25:
        risk_label = "Low Risk"
        risk_color = "#4CAF50"
    elif total_score < 50:
        risk_label = "Moderate"
        risk_color = "#FFC107"
    elif total_score < 75:
        risk_label = "High"
        risk_color = "#FF9800"
    else:
        risk_label = "Very High"
        risk_color = "#F44336"

    return {
        'total': total_score,
        'sentiment_component': round(sentiment_score, 1),
        'pattern_component': round(pattern_score, 1),
        'tone_component': round(tone_score, 1),
        'engagement_component': round(low_engagement_score, 1),
        'label': risk_label,
        'color': risk_color
    }


def compute_conversation_health(df: pd.DataFrame) -> dict:
    avg_sentiment = df['sentiment_compound'].mean()
    positivity_score = min(40, max(0, avg_sentiment * 40 + 20))

    if len(df['speaker'].unique()) >= 2:
        speaker_counts = df['speaker'].value_counts()
        ratio = speaker_counts.min() / speaker_counts.max()
        balance_score = ratio * 30
    else:
        balance_score = 0

    if 'pattern_toxicity' in df.columns:
        toxicity_rate = (df['pattern_toxicity'] > 0).mean()
        toxicity_score = max(0, 30 - toxicity_rate * 100)
    else:
        toxicity_score = 30

    health_score = round(positivity_score + balance_score + toxicity_score, 1)
    return {
        'score': min(100, health_score),
        'label': 'Healthy' if health_score > 60 else 'Needs Attention' if health_score > 35 else 'Concerning'
    }
