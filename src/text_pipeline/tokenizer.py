"""
src/text_pipeline/tokenizer.py

Wraps the HuggingFace DeBERTa-v3 tokenizer so the rest of the pipeline
doesn't need to touch HuggingFace APIs directly. Takes the `cleaned_text`
column produced by preprocessing.py and turns it into input_ids +
attention_mask tensors ready for deberta_encoder.py.
"""

import logging
from typing import List, Union

import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_MAX_LENGTH = 256  # most social-media posts fit well within this; keeps memory/time in check on Colab


class DepressionTextTokenizer:
    """
    Thin wrapper around AutoTokenizer for microsoft/deberta-v3-base.

    Usage:
        tok = DepressionTextTokenizer()
        encoded = tok.encode(["i feel so empty today", "great day at the park"])
        # encoded["input_ids"], encoded["attention_mask"] -> torch tensors, shape (batch, max_length)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        logger.info(f"Loading tokenizer: {model_name} (this downloads ~2-5MB of vocab files the first time)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts: Union[str, List[str]]) -> dict:
        """
        Tokenizes one string or a list of strings.
        Returns a dict with input_ids and attention_mask as torch tensors,
        padded/truncated to self.max_length.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, input_ids: torch.Tensor) -> str:
        """Converts token IDs back to readable text — mainly useful for debugging."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def encode_dataframe(self, df, text_col: str = "cleaned_text"):
        """
        Tokenizes an entire pandas dataframe column at once.
        Returns the same dict structure as encode(), sized (len(df), max_length).

        Usage:
            tok = DepressionTextTokenizer()
            encoded = tok.encode_dataframe(train_df)
            # then pass encoded["input_ids"], encoded["attention_mask"] into a DataLoader/Dataset
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")
        texts = df[text_col].fillna("").astype(str).tolist()
        return self.encode(texts)


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = DepressionTextTokenizer()

    sample_texts = [
        "i do not know anymore i just feel so empty right now cannot sleep",
        "helped me a lot to be honest feeling better today",
    ]

    encoded = tok.encode(sample_texts)
    print("input_ids shape:", encoded["input_ids"].shape)
    print("attention_mask shape:", encoded["attention_mask"].shape)
    print()
    print("First sample decoded back:", tok.decode(encoded["input_ids"][0]))