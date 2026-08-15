"""
src/text_pipeline/deberta_encoder.py

Loads microsoft/deberta-v3-base and wraps it so it outputs contextual
embeddings per token. Takes the input_ids + attention_mask produced by
tokenizer.py and returns a tensor of shape (batch, seq_len, hidden_size)
that the BiLSTM+Attention layer (bilstm_attention.py) will consume next.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


class DebertaEncoder(nn.Module):
    """
    Wraps DeBERTa-v3-base as a PyTorch nn.Module so it can be plugged
    directly into the larger text-branch model (DeBERTa -> BiLSTM -> Attention).

    Usage:
        encoder = DebertaEncoder()
        embeddings = encoder(input_ids, attention_mask)
        # embeddings.shape == (batch_size, seq_len, hidden_size)  # hidden_size = 768 for deberta-v3-base
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, freeze_early_layers: bool = True, num_frozen_layers: int = 6):
        """
        freeze_early_layers: if True, freezes the first `num_frozen_layers` transformer
        layers so training only fine-tunes the later layers. This is the blueprint's
        recommended approach to fit Colab's free-tier GPU memory and speed up training,
        since deberta-v3-base has 12 layers total.
        """
        super().__init__()
        logger.info(f"Loading DeBERTa model: {model_name}")
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size

        if freeze_early_layers:
            self._freeze_layers(num_frozen_layers)

    def _freeze_layers(self, num_frozen_layers: int):
        """Freezes embeddings + the first N encoder layers, per the blueprint's memory-saving advice."""
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = False

        encoder_layers = self.deberta.encoder.layer
        total_layers = len(encoder_layers)
        num_frozen_layers = min(num_frozen_layers, total_layers)

        for i in range(num_frozen_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

        logger.info(f"Froze embeddings + first {num_frozen_layers}/{total_layers} DeBERTa layers.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — from tokenizer.py
            attention_mask: (batch, seq_len) — from tokenizer.py

        Returns:
            token_embeddings: (batch, seq_len, hidden_size) — contextual embedding per token,
            ready to feed into the BiLSTM layer.
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return token_embeddings

    def get_cls_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns just the [CLS]-equivalent first-token embedding
        per sequence, shape (batch, hidden_size). Useful for baseline comparisons
        (plain DeBERTa without BiLSTM+Attention) mentioned in the blueprint's
        evaluation section.
        """
        token_embeddings = self.forward(input_ids, attention_mask)
        return token_embeddings[:, 0, :]


# ---------------------------------------------------------------------------
# Quick manual test — chains tokenizer.py -> deberta_encoder.py end to end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tokenizer import DepressionTextTokenizer

    tok = DepressionTextTokenizer()
    encoder = DebertaEncoder()
    encoder.eval()

    sample_texts = [
        "i do not know anymore i just feel so empty right now cannot sleep",
        "helped me a lot to be honest feeling better today",
    ]

    encoded = tok.encode(sample_texts)

    with torch.no_grad():
        embeddings = encoder(encoded["input_ids"], encoded["attention_mask"])

    print("Token embeddings shape:", embeddings.shape)
    print("Hidden size:", encoder.hidden_size)

    cls_emb = encoder.get_cls_embedding(encoded["input_ids"], encoded["attention_mask"])
    print("CLS embedding shape:", cls_emb.shape)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")