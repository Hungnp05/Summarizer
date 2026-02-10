import torch
import torch.nn as nn
import math
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")

# Tokenizer T5 (thêm legacy=False để tắt cảnh báo legacy behaviour)
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size=32128, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding đơn giản (học được)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))

        # Bật batch_first=True để tắt cảnh báo nested_tensor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src, tgt: (batch, seq_len)
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]

        # Không cần permute vì batch_first=True
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        return self.fc_out(output)

    # Phương thức generate đơn giản (greedy decode)
    def generate(self, input_ids, max_length=150, num_beams=1):
        # Để đơn giản, dùng greedy decode (có thể nâng cấp sau)
        batch_size = input_ids.size(0)
        tgt = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
        for _ in range(max_length):
            logits = self(input_ids, tgt)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == tokenizer.eos_token_id).all():
                break
        return tgt

self_built_model = Seq2SeqTransformer()
self_built_model.to(DEVICE)
self_built_model.eval()

# Load nếu đã train
self_built_model.load_state_dict(torch.load("summarizer_self_built.pt", map_location=DEVICE))

def summarize_self_built(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs.input_ids.to(DEVICE)
    
    with torch.no_grad():
        output_ids = self_built_model.generate(input_ids, max_length=150)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Vector (mean pooling từ decoder output - ví dụ đơn giản)
    with torch.no_grad():
        tgt = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
        output = self_built_model(input_ids, tgt)
        vector = output.mean(dim=1).squeeze().cpu().numpy().tolist()[:10]
    
    length_ratio = (len(summary.split()) / len(text.split())) * 100 if text else 0
    
    return {
        "summary": summary,
        "score": f"{length_ratio:.2f}% (tỷ lệ độ dài summary so với input)",
        "vector": vector
    }