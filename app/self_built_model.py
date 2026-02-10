import torch
import torch.nn as nn
import math
import numpy as np

DEVICE = torch.device("cpu")

from transformers import T5Tokenizer  # Dùng tokenizer T5 cho summarization

tokenizer = T5Tokenizer.from_pretrained("t5-small")

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size=32128, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))  # Simple pos encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return self.fc_out(output)

self_built_model = Seq2SeqTransformer()
self_built_model.to(DEVICE)
self_built_model.eval()

# Load nếu có
# self_built_model.load_state_dict(torch.load(r"E:\code\NLP\summarizer_self_built.pt", map_location=DEVICE))

def summarize_self_built(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs.input_ids.to(DEVICE)
    
    # Generate summary (greedy decoding đơn giản)
    with torch.no_grad():
        output_ids = self_built_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Vector (mean pooling từ decoder output)
    with torch.no_grad():
        tgt = torch.zeros(1, 1).long().to(DEVICE)
        output = self_built_model(input_ids, tgt)
        vector = output.mean(dim=1).squeeze().cpu().numpy().tolist()[:10]
    
    length_ratio = (len(summary.split()) / len(text.split())) * 100 if text else 0
    
    return {
        "summary": summary,
        "score": f"{length_ratio:.2f}% (summary length ratio)",
        "vector": vector
    }