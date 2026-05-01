import re
import pandas as pd


def parse_whatsapp_chat(text: str) -> pd.DataFrame:
    """Tries WhatsApp format first, falls back to flexible parsing."""
    df = _try_whatsapp_format(text)
    if df.empty:
        df = _try_flexible_format(text)
    return df


def _try_whatsapp_format(text: str) -> pd.DataFrame:
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
    system_keywords = ['end-to-end encrypted', 'joined using', 'left', 'added', 'removed']
    mask = ~df['message'].str.lower().str.contains('|'.join(system_keywords), na=False)
    df = df[mask].reset_index(drop=True)
    df = df[~df['message'].isin(['<Media omitted>', 'image omitted', 'video omitted'])]
    df = df[df['message'].str.len() > 1].reset_index(drop=True)
    return df


def _try_flexible_format(text: str) -> pd.DataFrame:
    """
    Accepts any format where a name/label precedes a colon.
    Works with:
      - Alice: hello
      - [Alice] hello
      - Alice - hello
      - Me: hey / You: hi
      - Alice (10:30): message
    """
    messages = []
    # Pattern: optional brackets, a name (1-30 chars), optional time in parens, colon or dash, message
    pattern = r'^\[?([A-Za-z][A-Za-z0-9 _\-]{0,29}?)\]?\s*(?:\([\d:apmAPM ]+\))?\s*[:\-]\s*(.+)$'

    for i, line in enumerate(text.strip().split('\n')):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            speaker, message = match.groups()
            speaker = speaker.strip()
            message = message.strip()
            if speaker and message and len(message) > 0:
                messages.append({
                    'timestamp': f"2024-01-01 {i:05d}",
                    'speaker': speaker,
                    'message': message
                })
        else:
            # Continuation line — append to last message
            if messages and line:
                messages[-1]['message'] += ' ' + line

    return pd.DataFrame(messages) if messages else pd.DataFrame()


def parse_manual_input(text: str) -> pd.DataFrame:
    """Alias — routes through the same flexible parser."""
    return _try_flexible_format(text)


def get_speakers(df: pd.DataFrame) -> list:
    return df['speaker'].value_counts().index.tolist()[:2]