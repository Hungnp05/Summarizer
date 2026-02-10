# Summarizer FastAPI App - Ứng dụng tóm tắt văn bản

Dự án xây dựng một ứng dụng web sử dụng **FastAPI** để tóm tắt (summarization) đoạn văn bản dài. Ứng dụng có hai chế độ:

- **Trang 1**: Sử dụng mô hình pre-trained **BART-large-cnn** (từ Hugging Face) – chính xác cao, sẵn dùng ngay.
- **Trang 2**: Mô hình Transformer Seq2Seq tự xây dựng từ đầu và tự huấn luyện trên bộ dữ liệu **CNN/DailyMail** – nhằm mục đích học hỏi, so sánh hiệu suất và tùy chỉnh.

Ứng dụng hiển thị: tóm tắt, tỷ lệ độ dài summary so với input (score), và vector embedding (10 chiều đầu của mean pooling).

## Giới thiệu dự án

- **Mục tiêu**: Xây dựng hệ thống tóm tắt văn bản với hai cách tiếp cận:
  - Sử dụng mô hình pre-trained mạnh mẽ để có kết quả chất lượng cao.
  - Tự implement và train mô hình Transformer Seq2Seq để hiểu sâu cơ chế hoạt động, dù hiệu suất thấp hơn.
- **Công nghệ chính**:
  - Backend: FastAPI + Jinja2 templates
  - Mô hình pre-trained: facebook/bart-large-cnn
  - Mô hình tự build: Transformer Encoder-Decoder (d_model=256, 2 layers encoder/decoder)
  - Dataset train: CNN/DailyMail (subset để train nhanh trên CPU)
- **Tính năng**: Giao diện web đơn giản với form nhập văn bản dài, hiển thị summary, score và vector embedding.

## Cách cài đặt và chuẩn bị môi trường

### Yêu cầu hệ thống
- Python 3.10+
- Máy có GPU để train nhanh hơn, nhưng CPU vẫn chạy được.

### Các bước cài đặt
1. Clone hoặc tải dự án về máy.
2. Mở terminal, di chuyển vào thư mục project:
cd E:\code\summarizer_fastapi_app
3. Tạo virtual environment (khuyến nghị):
python -m venv venv
venv\Scripts\activate   # Windows
4. Cài đặt dependencies:
pip install -r requirements.txt

## Cách train mô hình tự build

1. Đảm bảo đã cài đầy đủ dependencies (bao gồm torch, transformers, datasets).
2. Chạy script train:
python train_self_built.py

3. Quá trình:
- Tự động tải dataset CNN/DailyMail (lần đầu có thể mất vài phút).
- Train trên subset 3000 mẫu (có thể tăng lên nếu máy mạnh).
- Mỗi epoch in loss trung bình.
- Sau khi hoàn thành, lưu file model: `summarizer_self_built.pt` (trong thư mục gốc).
4. Thời gian train ước tính (trên CPU i7-13650HX):
- 3000 mẫu, 3 epochs: ~20–30 phút.
- Nếu dùng GPU (sau khi cài torch+CUDA): nhanh gấp 5–10 lần.

## Cách chạy dự án

1. Chạy server FastAPI:
uvicorn app.main:app --reload

2. Mở trình duyệt truy cập:
- Trang chính (BART pre-trained): http://127.0.0.1:8000/
- Trang tự build: http://127.0.0.1:8000/self-built
3. Sử dụng:
- Nhập đoạn văn bản dài vào ô textarea.
- Nhấn nút "Tóm tắt" → hiển thị summary, score (tỷ lệ độ dài), và vector embedding.

**Lưu ý**: Nếu chưa train model tự build, kết quả trên trang /self-built sẽ random. Train trước và load file `.pt` trong `self_built_model.py` để có kết quả thực tế.

## Hiệu suất khi train model tự train

- **Dataset**: CNN/DailyMail (subset 3000 mẫu để test nhanh).
- **Model**: Seq2Seq Transformer (d_model=256, 2 encoder/decoder layers).
- **Hardware**: CPU Intel i7-13650HX (không dùng GPU).
- **Kết quả điển hình** (3 epochs):
- Loss giảm dần từ ~3.5–4.0 xuống ~2.0–2.5.
- Không có accuracy trực tiếp (vì là generation task), nhưng có thể đánh giá thủ công: summary ngắn gọn, giữ ý chính nhưng còn thiếu chi tiết so với BART.
- Thời gian: ~20–30 phút cho 3 epochs trên CPU.
- **So sánh với pre-trained**:
- BART: Summary chất lượng cao, mạch lạc, gần với con người.
- Self-built: Summary cơ bản, còn nhiều lỗi ngữ pháp và thiếu ý, nhưng đã học được pattern tóm tắt sau vài epochs.
- **Cải thiện tiềm năng**:
- Train full dataset (~287k mẫu) + GPU → hiệu suất tăng đáng kể.
- Tăng layers/d_model, thêm beam search trong generate → summary tốt hơn.

Dự án giúp hiểu rõ sự khác biệt giữa mô hình pre-trained mạnh mẽ và mô hình tự xây dựng – dù tự train vẫn còn hạn chế về chất lượng nhưng mang lại giá trị học thuật cao.

Nếu bạn gặp vấn đề khi chạy, hãy kiểm tra terminal và đảm bảo đã cài đầy đủ dependencies.

LƯU Ý: ở file self_built_model.py trước khi train hãy để comment dòng (74): self_built_model.load_state_dict(torch.load("summarizer_self_built.pt", map_location=DEVICE))

sau khi chạy file train_self_built.py thì mới mở comment rồi chạy uvicorn app.main:app --reload