```markdown
# Talk to Documents

Ứng dụng hỏi đáp tài liệu PDF/TXT dựa trên RAG (Retrieval-Augmented Generation).  
Hệ thống hỗ trợ trích dẫn nguồn trực tiếp từ trang PDF gốc, giúp người dùng dễ dàng kiểm chứng câu trả lời.

## Tính năng nổi bật

- **Trích dẫn nguồn minh bạch**: Mỗi câu trả lời kèm số tham chiếu (ví dụ `[2]`). Người dùng có thể nhấn để xem đúng trang PDF và đoạn văn bản được sử dụng.
- **Xử lý bảng biểu tốt**: Tự động chuyển bảng thành Markdown, giữ nguyên cấu trúc số liệu.
- **Trung thực với dữ liệu**: Khi thông tin không có trong tài liệu (đặc biệt là số liệu trong biểu đồ), hệ thống trả lời rõ ràng thay vì suy diễn.
- **Giao diện nghiên cứu**: Bố cục 3 cột (Cài đặt – Chat – Xem tài liệu) hỗ trợ đối chiếu nhanh.
- **Tùy chỉnh linh hoạt**: Người dùng có thể điều chỉnh số nguồn tối đa (Top-K) và ngưỡng độ tương đồng.

## Hai phiên bản triển khai

| Phiên bản | Mô tả | Phù hợp khi |
|-----------|-------|-------------|
| **Streamlit Cloud** | Ứng dụng Streamlit độc lập, không cần backend riêng | Demo nhanh, dùng cá nhân, triển khai lên Streamlit Cloud |
| **FastAPI + Qdrant** | API + Qdrant + giao diện Streamlit | Cần lập chỉ mục nền, theo dõi tiến độ, xử lý tài liệu lớn, lưu trữ lâu dài |

---

## 1. Streamlit Cloud

### Yêu cầu
- Python 3.11+
- Gemini API key hoặc OpenAI API key

### Chạy cục bộ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Nhập API key trong sidebar. Khi triển khai Streamlit Cloud, đặt key trong **Settings → Secrets**.

### Lưu ý
- Hỗ trợ PDF có lớp chữ chọn được và TXT UTF-8 (tối đa 20 MB).
- PDF scan cần OCR trước khi tải lên.
- Mỗi phiên giữ dữ liệu riêng; dữ liệu sẽ mất khi phiên kết thúc hoặc ứng dụng khởi động lại.
- Chỉ đổi nhà cung cấp AI sau khi đã xóa tài liệu trong phiên (vector của các mô hình embedding không tương thích).

---

## 2. FastAPI + Qdrant

Phiên bản đầy đủ hơn, hỗ trợ:
- Lập chỉ mục nền có theo dõi tiến độ
- Phân tích bố cục PDF, bảng biểu
- Xử lý hình ảnh/biểu đồ bằng Gemini Vision (khi có API key)
- Truy xuất child-to-parent và trích dẫn vị trí trang

### Cài đặt

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-backend.txt
cp .env.example .env
```

Chạy Qdrant:

```bash
docker run -d --name qdrant-genai \
  -p 127.0.0.1:6333:6333 \
  -v qdrant_genai_data:/qdrant/storage \
  qdrant/qdrant
```

Khởi động dịch vụ (hai terminal):

```bash
# Terminal 1 – API
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 – Giao diện
streamlit run client.py
```

- Chat: http://127.0.0.1:8501  
- API docs: http://127.0.0.1:8000/docs  

### Cấu hình quan trọng (`.env`)

| Biến | Mô tả |
|------|------|
| `EMBEDDING_PROVIDER` | `local` (mặc định), `fastembed` hoặc `openai` |
| `LLM_PROVIDER` | `local`, `gemini` hoặc `openai` |
| `GEMINI_API_KEY` | Cần cho Gemini Vision và LLM Gemini |
| `OPENAI_API_KEY` | Cần khi dùng OpenAI |
| `QDRANT_URL` | Địa chỉ Qdrant (mặc định `http://127.0.0.1:6333`) |
| `MAX_UPLOAD_BYTES` | Giới hạn dung lượng file |

### Luồng sử dụng

1. Tải PDF/TXT (tối đa 20 MB mặc định).
2. Hệ thống trả job ID ngay và hiển thị tiến độ lập chỉ mục.
3. Sau khi tài liệu ở trạng thái `completed`, bắt đầu hỏi đáp.
4. Nhấn trích dẫn để xem trang PDF gốc và vùng được đánh dấu.

---

## Hạn chế hiện tại

- **Chưa đọc được số liệu trong biểu đồ/hình ảnh**: Hệ thống chủ yếu xử lý lớp văn bản. Hình ảnh và biểu đồ cần Gemini Vision (và API key) để mô tả; nếu không có sẽ bỏ qua nội dung bên trong.
- **PDF scan**: Cần OCR trước khi tải lên phiên bản Cloud.
- **Dữ liệu phiên (Cloud)**: Dữ liệu chỉ tồn tại trong phiên hiện tại, mất khi tải lại trang.
- **Giới hạn dung lượng**: Mặc định 20 MB mỗi file.
- **Phụ thuộc hạn mức API**: Khi dùng key miễn phí/giới hạn thấp, file dài có thể bị chậm do rate limit.

---

## Bảo mật

- Backend hiện **chưa có xác thực**. Chỉ nên chạy trên localhost hoặc bổ sung lớp xác thực trước khi mở ra mạng công cộng.
- Không commit API key, file `.env`, tài liệu riêng tư hoặc dữ liệu trong `uploads/` và `var/` lên GitHub.
```
