# ViABSA — Phân tích cảm xúc theo khía cạnh cho review tiếng Việt

Với mỗi review son môi tiếng Việt, mô hình dự đoán cảm xúc (không nhắc tới / tiêu cực / trung tính /
tích cực) cho **8 khía cạnh**: độ bền màu, chất son, mùi, giá, màu sắc, giao hàng, đóng gói, khác — và
có thể chỉ ra **những từ** ảnh hưởng nhất đến từng dự đoán (Integrated Gradients).

## Có gì trong dự án

* **Dữ liệu:** đọc CSV đúng cách (review có xuống dòng), chuẩn hóa văn bản, loại trùng lặp, phát hiện nhãn mâu thuẫn, **loại rò rỉ giữa train/val/test**.
* **Mô hình:** baseline TF-IDF + Logistic Regression; transformer đa đầu (PhoBERT / XLM-R / ViSoBERT, 8 đầu × 4 lớp).
* **Huấn luyện:** class weights, focal loss tùy chọn, warm-up, early stopping, AMP, MLflow tùy chọn.
* **Đánh giá:** macro-F1 theo khía cạnh, F1 phát hiện và F1 cảm xúc, ECE, phân tích theo nhóm, ma trận nhầm lẫn.
* **Triển khai:** ONNX int8, FastAPI, Streamlit (4 tab), Docker, GitHub Actions, pre-commit, 40 test.

## Kết quả baseline (tập test sau làm sạch, 1.534 review)

Mean macro-F1 **0,765** · F1 phát hiện khía cạnh 0,949 · F1 cảm xúc 0,709. Kết quả của các mô hình
transformer: điền vào bảng trong `README.md` sau khi huấn luyện.

Lưu ý: `price` và `packing` gần như không có mẫu tiêu cực/trung tính (test chỉ có 6 và 10 mẫu tiêu cực),
nên F1 của hai khía cạnh này kém tin cậy.

## Chạy nhanh

```bash
python -m venv .venv && source .venv/bin/activate
make install
# đặt data_train.csv, data_val.csv, data_test.csv vào data/raw/
make data && make baseline
make app        # Streamlit  http://localhost:8501
make api        # FastAPI    http://localhost:8000/docs
```

Huấn luyện transformer: `make install-train && make train-phobert` (nên dùng GPU). Chi tiết xem `README.md`.
