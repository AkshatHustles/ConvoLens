def generate_report(df, speaker_sentiment, speaker_patterns, risk_scores, health) -> str:
    lines = []
    lines.append("## ConvoLens Behavioral Analysis Report")
    lines.append(f"\n**Conversation health:** {health['label']} ({health['score']}/100)")
    lines.append(f"**Total messages analyzed:** {len(df)}")
    lines.append(f"**Participants:** {', '.join(df['speaker'].unique()[:2])}\n")
    lines.append("---")

    for speaker in df['speaker'].unique()[:2]:
        lines.append(f"\n### {speaker}")
        sentiment = speaker_sentiment.get(speaker, {})
        risk = risk_scores.get(speaker, {})
        patterns = speaker_patterns.get(speaker, {})

        lines.append(f"**Behavioral Risk Score:** {risk.get('total', 0)}/100 — *{risk.get('label', 'N/A')}*")

        avg = sentiment.get('avg_compound', 0)
        if avg > 0.2:
            sentiment_desc = "generally positive and warm"
        elif avg > 0:
            sentiment_desc = "slightly positive but reserved"
        elif avg > -0.2:
            sentiment_desc = "relatively neutral, with some tension"
        else:
            sentiment_desc = "predominantly negative in tone"

        lines.append(f"**Emotional Tone:** Communication is {sentiment_desc} (avg sentiment: {avg:+.2f})")
        lines.append(f"**Emotional Volatility:** {sentiment.get('emotional_volatility', 0):.2f} " +
                     ("(high fluctuation detected)" if sentiment.get('emotional_volatility', 0) > 0.4 else "(relatively stable)"))

        top_patterns = sorted(
            [(k, v['occurrence_count']) for k, v in patterns.items() if v['occurrence_count'] > 0],
            key=lambda x: x[1], reverse=True
        )[:3]

        if top_patterns:
            lines.append(f"**Dominant behavioral patterns:** " +
                         ', '.join([f"{p[0].replace('_', ' ')} ({p[1]}x)" for p in top_patterns]))
        else:
            lines.append("**Dominant behavioral patterns:** None significantly detected")

        speaker_df = df[df['speaker'] == speaker]
        lines.append(f"**Avg message length:** {speaker_df['word_count'].mean():.1f} words")
        lines.append(f"**Total messages:** {len(speaker_df)} ({len(speaker_df)/len(df)*100:.0f}% of conversation)")

    lines.append("\n---")
    lines.append("\n> **Ethical Note:** This analysis identifies behavioral and linguistic patterns using NLP techniques only. "
                 "It does not scientifically determine intent, truthfulness, or relationship quality. "
                 "Results should be interpreted as conversational observations, not conclusions.")

    return '\n'.join(lines)
