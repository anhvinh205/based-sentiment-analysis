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

### Windows (terminal trong VS Code / Command Prompt)

Cần Python 3.11+ và Git. Repository không chứa bộ dữ liệu review hoặc model đã huấn luyện. Hãy lấy
dữ liệu từ nguồn cho phép bạn sử dụng, đặt ba file chia tập vào `data\raw\`, và đọc
[DATA_STATEMENT.md](DATA_STATEMENT.md) trước khi chia sẻ dữ liệu.

Chạy các lệnh sau trong terminal Command Prompt:

```cmd
git clone https://github.com/anhvinh205/based-sentiment-analysis.git
cd based-sentiment-analysis
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,serve,app,onnx]"
```

Chép `data_train.csv`, `data_val.csv`, `data_test.csv` vào `data\raw\`, sau đó tiền xử lý dữ liệu và
huấn luyện baseline:

```cmd
python -m viabsa.data.prepare --config configs/data.yaml
python -m viabsa.training.train_baseline --config configs/baseline.yaml
```

Model được tạo tại `models\baseline`. Mở hai terminal VS Code riêng (mỗi terminal đều kích hoạt
`.venv`) để chạy API và giao diện:

```cmd
set VIABSA_MODEL_PATH=models/baseline
uvicorn viabsa.serving.api:app --reload --port 8000
```

```cmd
set VIABSA_MODEL_PATH=models/baseline
streamlit run app/streamlit_app.py
```

Mở giao diện tại `http://localhost:8501`; tài liệu API tại `http://localhost:8000/docs`.

### Linux / macOS

```bash
git clone https://github.com/anhvinh205/based-sentiment-analysis.git
cd based-sentiment-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,serve,app,onnx]"
# Đặt data_train.csv, data_val.csv, data_test.csv đã được phép sử dụng vào data/raw/
python -m viabsa.data.prepare --config configs/data.yaml
python -m viabsa.training.train_baseline --config configs/baseline.yaml
```

Sau đó mở hai terminal riêng, đặt biến `VIABSA_MODEL_PATH=models/baseline` trong mỗi terminal và
chạy lệnh API/UI tương ứng ở trên.

### Docker

Trước tiên cần chuẩn bị dữ liệu và huấn luyện baseline như hướng dẫn trên. Docker Compose mount thư
mục `models/` từ máy host; model không nằm sẵn trong Docker image.

```cmd
docker compose up --build
```

API: `http://localhost:8000/docs`; giao diện: `http://localhost:8501`.

### Huấn luyện transformer (tùy chọn)

```bash
python -m pip install -e ".[train]"
python -m viabsa.training.train_transformer --config configs/transformer_phobert.yaml
```

Có thể thay file cấu hình bằng `configs/transformer_xlmr.yaml` hoặc
`configs/transformer_visobert.yaml`. Nên dùng GPU tương thích CUDA. Hướng dẫn chi tiết hơn ở
[README tiếng Anh](README.md).
