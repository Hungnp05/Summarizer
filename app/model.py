from transformers import pipeline

# Load pre-trained summarizer (BART-large-cnn)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    # Tóm tắt văn bản
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    # Vector embedding: Sử dụng mean pooling từ BART hidden states (cần forward đầy đủ)
    # Để đơn giản, dùng sentence-transformers cho vector đại diện
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    vector = embedder.encode(summary).tolist()[:10]
    
    # "Score": Độ dài summary so với input (tỷ lệ %)
    length_ratio = (len(summary.split()) / len(text.split())) * 100 if text else 0
    
    return {
        "summary": summary,
        "score": f"{length_ratio:.2f}% (summary length ratio)",
        "vector": vector
    }