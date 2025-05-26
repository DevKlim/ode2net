# File: models/text_encoder.py
import torch
from transformers import AutoTokenizer, AutoModel

class TextEncoder:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output  # (1, hidden_size)
        return embedding.squeeze(0)  # (hidden_size,)
