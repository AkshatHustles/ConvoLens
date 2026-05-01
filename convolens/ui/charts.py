import plotly.graph_objects as go
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.keywords import BEHAVIORAL_PATTERNS


def sentiment_timeline_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ['#5C6BC0', '#EF5350']
    for i, speaker in enumerate(df['speaker'].unique()[:2]):
        speaker_df = df[df['speaker'] == speaker].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=list(range(len(speaker_df))),
            y=speaker_df['sentiment_compound'],
            name=speaker,
            mode='lines+markers',
            line=dict(color=colors[i % 2], width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{speaker}</b><br>Message #%{{x}}<br>Sentiment: %{{y:.2f}}<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hrect(y0=-1, y1=-0.05, fillcolor="red", opacity=0.05)
    fig.add_hrect(y0=0.05, y1=1, fillcolor="green", opacity=0.05)
    fig.update_layout(
        title="Sentiment Timeline",
        xaxis_title="Message sequence",
        yaxis_title="Sentiment score",
        yaxis=dict(range=[-1.1, 1.1]),
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def behavioral_radar_chart(speaker_patterns: dict) -> go.Figure:
    categories = list(BEHAVIORAL_PATTERNS.keys())
    categories_clean = [c.replace('_', ' ').title() for c in categories]
    fig = go.Figure()
    colors = ['rgba(92, 107, 192, 0.3)', 'rgba(239, 83, 80, 0.3)']
    line_colors = ['#5C6BC0', '#EF5350']
    for i, (speaker, patterns) in enumerate(list(speaker_patterns.items())[:2]):
        values = [patterns.get(cat, {}).get('total_score', 0) for cat in categories]
        max_val = max(values) if max(values) > 0 else 1
        values_norm = [round(v / max_val * 10, 2) for v in values]
        fig.add_trace(go.Scatterpolar(
            r=values_norm + [values_norm[0]],
            theta=categories_clean + [categories_clean[0]],
            fill='toself',
            name=speaker,
            fillcolor=colors[i % 2],
            line=dict(color=line_colors[i % 2], width=2)
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title="Behavioral Pattern Radar",
        showlegend=True,
        height=400
    )
    return fig


def speaker_contribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df['speaker'].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=['#5C6BC0', '#EF5350']),
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Messages: %{value}<br>Share: %{percent}<extra></extra>"
    ))
    fig.update_layout(title="Message Contribution", height=320, showlegend=False)
    return fig


def toxicity_trend_chart(df: pd.DataFrame) -> go.Figure:
    toxic_df = df[df['pattern_toxicity'] > 0].copy()
    fig = go.Figure()
    for speaker in df['speaker'].unique()[:2]:
        speaker_toxic = toxic_df[toxic_df['speaker'] == speaker].reset_index()
        if not speaker_toxic.empty:
            fig.add_trace(go.Bar(
                x=speaker_toxic['index'],
                y=speaker_toxic['pattern_toxicity'],
                name=speaker,
                hovertemplate="<b>%{text}</b><extra></extra>",
                text=speaker_toxic['message'].str[:40] + '...'
            ))
    fig.update_layout(
        title="Toxicity Signal Locations",
        xaxis_title="Message index",
        yaxis_title="Toxicity score",
        template="plotly_white",
        height=300,
        barmode='stack'
    )
    return fig


def emotional_volatility_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ['#5C6BC0', '#EF5350']
    for i, speaker in enumerate(df['speaker'].unique()[:2]):
        speaker_series = df[df['speaker'] == speaker]['sentiment_compound'].reset_index(drop=True)
        rolling_std = speaker_series.rolling(window=5, min_periods=2).std()
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_std))),
            y=rolling_std,
            name=speaker,
            fill='tozeroy',
            line=dict(color=colors[i % 2], width=1.5)
        ))
    fig.update_layout(
        title="Emotional Volatility Over Time",
        xaxis_title="Message sequence",
        yaxis_title="Sentiment std deviation (rolling)",
        template="plotly_white",
        height=300
    )
    return fig


def risk_gauge(score: float, speaker: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': f"{speaker}<br><span style='font-size:0.8em'>Behavioral Risk</span>"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#5C6BC0"},
            'steps': [
                {'range': [0, 25], 'color': '#C8E6C9'},
                {'range': [25, 50], 'color': '#FFF9C4'},
                {'range': [50, 75], 'color': '#FFE0B2'},
                {'range': [75, 100], 'color': '#FFCDD2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=20))
    return fig
