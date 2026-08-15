"""
src/text_pipeline/preprocessing.py

Cleans raw social-media text (Reddit/Twitter posts) before it goes into the
DeBERTa-v3 tokenizer. Works on any dataset that has a `text` column and a
`label` column (0 = non-depressed, 1 = depressed) — so it's dataset-agnostic
across RSDD, eRisk, and the Twitter Kaggle dataset.

Pipeline order matters:
    1. Lowercase
    2. Remove URLs / mentions / markdown-ish junk
    3. Convert emojis to text        (":)" -> "smiling face")
    4. Expand slang/abbreviations    ("idk" -> "i do not know")
    5. Expand contractions           ("don't" -> "do not")
    6. Strip extra whitespace / punctuation noise
"""

import re
import json
import logging
from pathlib import Path

import pandas as pd
import emoji
import contractions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slang / abbreviation dictionary
# ---------------------------------------------------------------------------
# This is the starter set. The full dictionary should live in
# data/external/slang_emoji_dictionaries/slang.json — if that file exists,
# it's loaded and merged on top of this default set (file wins on conflicts).
DEFAULT_SLANG_DICT = {
    "idk": "i do not know",
    "idc": "i do not care",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "irl": "in real life",
    "omg": "oh my god",
    "btw": "by the way",
    "rn": "right now",
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "im": "i am",
    "dont": "do not",
    "cant": "cannot",
    "wont": "will not",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "fml": "fuck my life",
    "nvm": "never mind",
    "afaik": "as far as i know",
    "asap": "as soon as possible",
    "brb": "be right back",
    "ftw": "for the win",
    "ngl": "not gonna lie",
    "tbf": "to be fair",
    "ppl": "people",
    "bc": "because",
    "b4": "before",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",
}


def load_slang_dict(external_path: str = "data/external/slang_emoji_dictionaries/slang.json") -> dict:
    """
    Loads the project's slang dictionary. Falls back to DEFAULT_SLANG_DICT
    if the external file doesn't exist yet (e.g. no one has added it to
    data/external/ yet — the pipeline should never crash because of that).
    """
    slang_dict = dict(DEFAULT_SLANG_DICT)
    path = Path(external_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                custom = json.load(f)
            slang_dict.update(custom)
            logger.info(f"Loaded {len(custom)} custom slang entries from {external_path}")
        except Exception as e:
            logger.warning(f"Could not load {external_path}: {e}. Using default slang dict only.")
    else:
        logger.info(f"No custom slang file found at {external_path}, using {len(slang_dict)} default entries.")
    return slang_dict


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
SUBREDDIT_PATTERN = re.compile(r"r/\w+")
HTML_PATTERN = re.compile(r"<.*?>")
EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s.,!?']")


def remove_urls_mentions(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = SUBREDDIT_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    return text


def convert_emojis(text: str) -> str:
    """Turns emoji characters into readable text, e.g. '😢' -> 'crying face'."""
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ").replace(":", " ")
    return text


def expand_slang(text: str, slang_dict: dict) -> str:
    words = text.split()
    expanded = [slang_dict.get(w, w) for w in words]
    return " ".join(expanded)


def expand_contractions_step(text: str) -> str:
    try:
        return contractions.fix(text)
    except Exception:
        return text


def strip_noise(text: str) -> str:
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = EXTRA_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_text(raw_text: str, slang_dict: dict) -> str:
    """
    Runs one piece of raw text through the full cleaning pipeline.
    Order matters — emoji conversion and slang expansion must happen
    before punctuation stripping, or you'd lose the words they produce.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    text = raw_text.lower()
    text = remove_urls_mentions(text)
    text = convert_emojis(text)
    text = expand_slang(text, slang_dict)
    text = expand_contractions_step(text)
    text = strip_noise(text)
    return text


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    slang_dict: dict | None = None,
    min_word_count: int = 3,
) -> pd.DataFrame:
    """
    Cleans an entire dataframe of posts.

    Expects columns:
        text_col  - raw post text
        label_col - 0 (non-depressed) / 1 (depressed)

    Drops rows that end up empty or too short after cleaning (these add
    noise, not signal, to the model).
    """
    if slang_dict is None:
        slang_dict = load_slang_dict()

    if text_col not in df.columns:
        raise ValueError(f"Expected a '{text_col}' column, got columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Expected a '{label_col}' column, got columns: {list(df.columns)}")

    logger.info(f"Starting preprocessing on {len(df)} rows...")

    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(lambda t: clean_text(t, slang_dict))

    before = len(df)
    df["word_count"] = df["cleaned_text"].str.split().str.len().fillna(0)
    df = df[df["word_count"] >= min_word_count].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with fewer than {min_word_count} words after cleaning.")

    df = df.drop(columns=["word_count"])
    logger.info(f"Preprocessing complete. {len(df)} rows remain.")
    return df


def preprocess_csv(
    input_path: str,
    output_path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> None:
    """
    Convenience wrapper: reads a raw CSV, cleans it, writes the cleaned CSV.
    Use this from train_text_model.py or a notebook, e.g.:

        preprocess_csv(
            "data/raw/text/rsdd/train.csv",
            "data/processed/text/train.csv",
        )
    """
    logger.info(f"Reading {input_path}")
    df = pd.read_csv(input_path)

    slang_dict = load_slang_dict()
    cleaned_df = preprocess_dataframe(df, text_col=text_col, label_col=label_col, slang_dict=slang_dict)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")


# ---------------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check the pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_posts = [
        "idk anymore, i just feel so empty rn :( cant sleep",
        "Check this out! https://example.com @friend loved it lol 😂😂",
        "r/depression helped me a lot tbh, feeling better today :)",
        "",
        "ok",
    ]
    slang_dict = load_slang_dict()
    for post in sample_posts:
        print(f"RAW:     {post!r}")
        print(f"CLEANED: {clean_text(post, slang_dict)!r}")
        print("-" * 60)