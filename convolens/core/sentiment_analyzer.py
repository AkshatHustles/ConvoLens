from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_scores(text: str) -> dict:
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    scores['label'] = sentiment
    return scores


def analyze_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sentiment_data = df['message'].apply(get_sentiment_scores)
    df['sentiment_compound'] = sentiment_data.apply(lambda x: x['compound'])
    df['sentiment_pos'] = sentiment_data.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiment_data.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_data.apply(lambda x: x['neu'])
    df['sentiment_label'] = sentiment_data.apply(lambda x: x['label'])
    return df


def get_speaker_sentiment_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        summary[speaker] = {
            'avg_compound': round(speaker_df['sentiment_compound'].mean(), 3),
            'positive_msgs': int((speaker_df['sentiment_label'] == 'positive').sum()),
            'negative_msgs': int((speaker_df['sentiment_label'] == 'negative').sum()),
            'neutral_msgs': int((speaker_df['sentiment_label'] == 'neutral').sum()),
            'total_msgs': len(speaker_df),
            'emotional_volatility': round(speaker_df['sentiment_compound'].std(), 3)
        }
    return summary
