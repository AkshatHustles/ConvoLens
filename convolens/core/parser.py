import re
import pandas as pd


def parse_whatsapp_chat(text: str) -> pd.DataFrame:
    """
    Parses WhatsApp exported .txt chat into a structured DataFrame.
    Handles both 12-hour and 24-hour timestamp formats.
    Returns: DataFrame with columns [timestamp, speaker, message]
    """
    pattern = r'\[?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]?\s*[-\u2013]?\s*([^:]+):\s*(.*)'

    messages = []
    current_msg = None

    for line in text.strip().split('\n'):
        match = re.match(pattern, line)
        if match:
            if current_msg:
                messages.append(current_msg)
            date_str, time_str, speaker, message = match.groups()
            current_msg = {
                'timestamp': f"{date_str} {time_str}",
                'speaker': speaker.strip(),
                'message': message.strip()
            }
        else:
            if current_msg and line.strip():
                current_msg['message'] += ' ' + line.strip()

    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)

    if df.empty:
        return df

    system_keywords = ['end-to-end encrypted', 'joined using', 'left',
                        'added', 'removed', 'changed the subject']
    mask = ~df['message'].str.lower().str.contains('|'.join(system_keywords), na=False)
    df = df[mask].reset_index(drop=True)

    df = df[~df['message'].isin(['<Media omitted>', 'image omitted', 'video omitted'])]
    df = df[df['message'].str.len() > 1].reset_index(drop=True)

    return df


def parse_manual_input(text: str) -> pd.DataFrame:
    """
    Parses simple manual input format:
    Person A: message text
    Person B: message text
    """
    lines = [l.strip() for l in text.strip().split('\n') if ':' in l and l.strip()]
    messages = []

    for i, line in enumerate(lines):
        colon_idx = line.index(':')
        speaker = line[:colon_idx].strip()
        message = line[colon_idx+1:].strip()
        if speaker and message:
            messages.append({
                'timestamp': f"2024-01-01 {i:02d}:00",
                'speaker': speaker,
                'message': message
            })

    return pd.DataFrame(messages) if messages else pd.DataFrame()


def get_speakers(df: pd.DataFrame) -> list:
    """Returns list of unique speakers sorted by message count."""
    speakers = df['speaker'].value_counts().index.tolist()
    return speakers[:2]
