import torch
import torch.nn as nn
import math
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")

# Tokenizer T5
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size=32128, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding đơn giản
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))

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
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        return self.fc_out(output)

    def generate(self, input_ids, max_length=150, repetition_penalty=1.2):
        batch_size = input_ids.size(0)
        tgt = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
        generated = tgt.clone()
        generated_tokens = set()

        for step in range(max_length):
            logits = self(input_ids, generated)
            next_token_logits = logits[:, -1, :]

            # Repetition penalty
            for token in generated_tokens:
                next_token_logits[0, token] /= repetition_penalty

            # Chọn token tiếp theo
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            generated_tokens.add(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

        return generated

# Khởi tạo model
self_built_model = Seq2SeqTransformer()
self_built_model.to(DEVICE)
self_built_model.eval()

# Load weights đã train
try:
    state_dict = torch.load(r"E:\code\summarizer_fastapi_app\summarizer_self_built.pt", map_location=DEVICE)
    self_built_model.load_state_dict(state_dict)
    print("Loaded trained weights from summarizer_self_built.pt successfully!")
    print("Sample weight from fc_out layer:", self_built_model.fc_out.weight[0][:5])
except FileNotFoundError:
    print("File summarizer_self_built.pt not found!")
except Exception as e:
    print(f"Failed to load weights: {str(e)}")
    print("Model is running with random weights - summary may be meaningless!")

def summarize_self_built(text):
    """
    Hàm tóm tắt văn bản bằng model self-built.
    """
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs.input_ids.to(DEVICE)
    
    with torch.no_grad():
        output_ids = self_built_model.generate(input_ids, max_length=150, repetition_penalty=1.2)
    
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Nếu summary rỗng hoặc chỉ lặp token, trả fallback
    if not summary.strip() or len(summary.split()) < 3:
        summary = "[Model chưa sinh được tóm tắt hợp lý - thử train thêm hoặc kiểm tra weights]"

    # Vector embedding
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