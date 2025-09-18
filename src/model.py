import torch
import torch.nn as nn
from transformers import AutoModel

class CustomDistilBERT(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Mean pooling across tokens
        pooled = torch.mean(last_hidden, dim=1)

        # Dropout + classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits
