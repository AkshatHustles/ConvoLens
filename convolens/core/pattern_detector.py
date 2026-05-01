import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.keywords import BEHAVIORAL_PATTERNS


def detect_patterns_in_message(message: str) -> dict:
    message_lower = message.lower()
    detected = {}
    for pattern_name, config in BEHAVIORAL_PATTERNS.items():
        score = 0
        matched_keywords = []
        for keyword in config['keywords']:
            if keyword.lower() in message_lower:
                score += config['weight']
                matched_keywords.append(keyword)
        if score > 0:
            detected[pattern_name] = {
                'score': round(score, 2),
                'matched': matched_keywords,
                'weight': config['weight']
            }
    return detected


def analyze_patterns_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pattern_results = df['message'].apply(detect_patterns_in_message)
    for pattern_name in BEHAVIORAL_PATTERNS.keys():
        df[f'pattern_{pattern_name}'] = pattern_results.apply(
            lambda x: x.get(pattern_name, {}).get('score', 0)
        )
    df['pattern_total_score'] = df[[f'pattern_{p}' for p in BEHAVIORAL_PATTERNS.keys()]].sum(axis=1)
    df['patterns_detected'] = pattern_results.apply(lambda x: list(x.keys()))
    return df


def get_speaker_pattern_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        pattern_scores = {}
        for pattern_name in BEHAVIORAL_PATTERNS.keys():
            col = f'pattern_{pattern_name}'
            total = speaker_df[col].sum()
            count = (speaker_df[col] > 0).sum()
            pattern_scores[pattern_name] = {
                'total_score': round(float(total), 2),
                'occurrence_count': int(count),
                'frequency_pct': round(count / len(speaker_df) * 100, 1)
            }
        summary[speaker] = pattern_scores
    return summary
