import re
import pandas as pd


def parse_whatsapp_chat(text: str) -> pd.DataFrame:
    """
    Parses WhatsApp exported chats.
    Supports multiple Android/iPhone formats.
    """

    lines = text.splitlines()
    data = []

    patterns = [

        # Android style
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[APMapm]{0,2})\s-\s([^:]+):\s(.+)$",

        # iPhone style
        r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[APMapm]{0,2})\]\s([^:]+):\s(.+)$"
    ]

    for line in lines:

        matched = False

        for pattern in patterns:

            match = re.match(pattern, line)

            if match:
                date, time, speaker, message = match.groups()

                data.append({
                    "date": date,
                    "time": time,
                    "speaker": speaker.strip(),
                    "message": message.strip()
                })

                matched = True
                break

        # Handle multiline messages
        if not matched and data:
            data[-1]["message"] += " " + line.strip()

    return pd.DataFrame(data)


def parse_manual_input(text: str) -> pd.DataFrame:
    """
    Flexible manual conversation parser.
    """

    lines = text.splitlines()
    data = []

    patterns = [

        r"^([^:]+):\s(.+)$",                 # Alice: Hello
        r"^\[([^\]]+)\]\s(.+)$",             # [Alice] Hello
        r"^([^-]+)-\s(.+)$",                 # Alice - Hello
        r"^(.+?)\s\(\d{1,2}:\d{2}\):\s(.+)$" # Alice (10:30): Hello
    ]

    for line in lines:

        line = line.strip()

        if not line:
            continue

        matched = False

        for pattern in patterns:

            match = re.match(pattern, line)

            if match:
                speaker, message = match.groups()

                data.append({
                    "speaker": speaker.strip(),
                    "message": message.strip()
                })

                matched = True
                break

        if not matched:
            data.append({
                "speaker": "Unknown",
                "message": line
            })

    return pd.DataFrame(data)


def get_speakers(df: pd.DataFrame):
    return df['speaker'].dropna().unique().tolist()