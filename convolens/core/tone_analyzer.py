import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def detect_tone_shifts(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    df = df.copy()
    df['tone_shift_score'] = 0.0
    df['is_tone_shift'] = False

    messages = df['processed_message'].fillna('').tolist()

    if len(messages) < window_size * 2:
        return df

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))

    try:
        tfidf_matrix = vectorizer.fit_transform(messages)
    except Exception:
        return df

    shift_scores = [0.0] * len(messages)

    for i in range(window_size, len(messages) - window_size):
        before = tfidf_matrix[max(0, i-window_size):i]
        after = tfidf_matrix[i:min(len(messages), i+window_size)]
        before_avg = np.asarray(before.mean(axis=0))
        after_avg = np.asarray(after.mean(axis=0))
        if before_avg.sum() > 0 and after_avg.sum() > 0:
            sim = cosine_similarity(before_avg, after_avg)[0][0]
            shift_scores[i] = round(1 - sim, 3)

    df['tone_shift_score'] = shift_scores
    df['is_tone_shift'] = df['tone_shift_score'] > 0.65

    return df


def compute_tfidf_keywords(df: pd.DataFrame) -> dict:
    keywords_per_speaker = {}
    for speaker in df['speaker'].unique():
        speaker_text = ' '.join(df[df['speaker'] == speaker]['processed_message'].fillna(''))
        other_text = ' '.join(df[df['speaker'] != speaker]['processed_message'].fillna(''))
        if not speaker_text.strip():
            continue
        try:
            vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
            corpus = [speaker_text, other_text]
            tfidf = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf[0].toarray()[0]
            keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            keywords_per_speaker[speaker] = keyword_scores[:15]
        except Exception:
            keywords_per_speaker[speaker] = []
    return keywords_per_speaker
