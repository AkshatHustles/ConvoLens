import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

PRESERVE_WORDS = {'not', 'no', 'never', 'always', 'very', 'but', 'however',
                   'although', 'fine', 'ok', 'okay', 'sure', 'whatever'}
EFFECTIVE_STOPWORDS = STOP_WORDS - PRESERVE_WORDS


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    return word_tokenize(clean_text(text))


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in EFFECTIVE_STOPWORDS and len(t) > 1]


def lemmatize_tokens(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


def full_preprocess(text: str) -> str:
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


def preprocess_dataframe(df):
    df = df.copy()
    df['cleaned_message'] = df['message'].apply(clean_text)
    df['processed_message'] = df['message'].apply(full_preprocess)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    return df
