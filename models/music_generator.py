# File: models/music_generator.py
import torch
import torch.nn as nn

class MusicGenerator(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super(MusicGenerator, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_size, num_tokens)

    def forward(self, emotion_embedding: torch.Tensor, music_tokens: torch.Tensor) -> torch.Tensor:
        """
        emotion_embedding: (batch_size, embedding_size)
        music_tokens:      (seq_len, batch_size)
        returns logits:    (seq_len, batch_size, num_tokens)
        """
        src = emotion_embedding.unsqueeze(0)  # (1, batch, E)
        tgt = self.embedding(music_tokens)   # (T, batch, E)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate(
        self,
        emotion_embedding: torch.Tensor,
        start_token: int,
        max_length: int = 3000,
        temperature: float = 1.2
    ) -> list[int]:
        self.eval()
        device = emotion_embedding.device
        generated: list[int] = [start_token]
        with torch.no_grad():
            for _ in range(max_length - 1):
                seq = torch.tensor(generated, device=device).unsqueeze(1)  # (T,1)
                logits = self.forward(emotion_embedding.unsqueeze(0), seq)  # (T,1,C)
                probs = torch.softmax(logits[-1] / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
                generated.append(next_token)
        return generated
