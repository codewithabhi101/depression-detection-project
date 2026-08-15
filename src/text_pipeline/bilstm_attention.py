"""
src/text_pipeline/bilstm_attention.py

Takes DeBERTa's per-token contextual embeddings (batch, seq_len, hidden_size)
and passes them through:
    1. A bidirectional LSTM (captures sequential patterns DeBERTa's
       self-attention alone may under-weight, per the blueprint architecture)
    2. A word-level attention mechanism (learns which tokens matter most
       for the depressed/non-depressed decision, and doubles as the
       explainability signal - these attention weights are what LIME's
       word-highlighting will visualize)

Output: a single fixed-size vector per sequence, ready for either
    (a) a standalone text-only classifier head (this file provides one), or
    (b) the fusion layer (src/fusion/) which combines it with the face branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau-style additive attention over BiLSTM outputs.

    Learns a weight per token showing how much it contributed to the final
    prediction. These weights are exactly what src/explainability/lime_explainer.py
    and the word-highlight visualization will read from.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # hidden_size here is the BiLSTM output size (already *2 for bidirectional)
        self.attn_weights = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size) — BiLSTM output
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            context: (batch, hidden_size) — weighted sum of lstm_output, the
                      final sentence representation
            attn_weights: (batch, seq_len) — per-token importance scores,
                      sum to 1 across real tokens (padding gets ~0 weight)
        """
        # score each token
        scores = torch.tanh(self.attn_weights(lstm_output))       # (batch, seq_len, hidden_size)
        scores = self.context_vector(scores).squeeze(-1)          # (batch, seq_len)

        # mask out padding tokens before softmax so they get ~0 weight
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)                   # (batch, seq_len)

        # weighted sum of the LSTM outputs -> single vector per sequence
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_size)

        return context, attn_weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Full text-branch head: DeBERTa embeddings -> BiLSTM -> Attention -> classifier.

    This can be used standalone (text-only baseline, see the blueprint's
    baseline #3: "Text-only DeBERTa-v3 + BiLSTM + Attention, no fusion") or
    the `get_features()` method can be called from the fusion layer to get
    the pre-softmax vector instead of a final prediction.
    """

    def __init__(
        self,
        deberta_hidden_size: int = 768,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=deberta_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        bilstm_output_size = lstm_hidden_size * 2  # *2 because bidirectional
        self.attention = Attention(bilstm_output_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bilstm_output_size, num_classes)

        self.feature_size = bilstm_output_size  # exposed for fusion.py to read

    def get_features(self, deberta_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """
        Runs BiLSTM + Attention only, returns the pre-classifier feature vector
        and the attention weights (for explainability). This is what
        src/fusion/feature_fusion.py calls to get the text branch's contribution.

        Returns:
            features: (batch, feature_size)
            attn_weights: (batch, seq_len)
        """
        deberta_embeddings = deberta_embeddings.to(torch.float32)
        lstm_out, _ = self.bilstm(deberta_embeddings)          # (batch, seq_len, lstm_hidden*2)       # (batch, seq_len, lstm_hidden*2)
        features, attn_weights = self.attention(lstm_out, attention_mask)
        return features, attn_weights

    def forward(self, deberta_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """
        Full forward pass -> standalone text-only prediction.

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, seq_len) — for explainability/word highlighting
        """
        features, attn_weights = self.get_features(deberta_embeddings, attention_mask)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits, attn_weights

    def predict(self, deberta_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """Convenience method: returns predicted class + confidence + attention weights."""
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(deberta_embeddings, attention_mask)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        return pred_class, confidence, attn_weights


# ---------------------------------------------------------------------------
# Quick manual test — chains tokenizer -> deberta_encoder -> bilstm_attention
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tokenizer import DepressionTextTokenizer
    from deberta_encoder import DebertaEncoder

    tok = DepressionTextTokenizer()
    encoder = DebertaEncoder()
    encoder.eval()

    model = BiLSTMAttentionClassifier(deberta_hidden_size=encoder.hidden_size)
    model.eval()

    sample_texts = [
        "i do not know anymore i just feel so empty right now cannot sleep",
        "helped me a lot to be honest feeling better today",
    ]

    encoded = tok.encode(sample_texts)

    with torch.no_grad():
        deberta_embeddings = encoder(encoded["input_ids"], encoded["attention_mask"])

    pred_class, confidence, attn_weights = model.predict(deberta_embeddings, encoded["attention_mask"])

    print("Predicted classes:", pred_class.tolist(), "(0=non-depressed, 1=depressed — untrained, random for now)")
    print("Confidence:", confidence.tolist())
    print("Attention weights shape:", attn_weights.shape)
    print("Feature vector size (for fusion):", model.feature_size)

    # show which tokens got the most attention for the first sample
    tokens = tok.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    weights = attn_weights[0].tolist()
    top5 = sorted(zip(tokens, weights), key=lambda x: -x[1])[:5]
    print("\nTop 5 attended tokens (sample 1, untrained so not meaningful yet):")
    for token, weight in top5:
        print(f"  {token}: {weight:.4f}")