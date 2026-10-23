# Talk to Documents

Ứng dụng hỏi đáp tài liệu PDF/TXT bằng RAG. Repository có hai phiên bản chạy độc lập:

| Phiên bản | Mã nguồn | Cách chạy | Lưu trữ |
| --- | --- | --- | --- |
| [Streamlit Cloud](https://chatbot-ni241maz32f.streamlit.app/) | `streamlit_app.py`, `cloud_rag.py`, `providers.py` | Một ứng dụng Streamlit, không cần FastAPI | Tài liệu và Qdrant local trong phiên Streamlit |
| FastAPI + Qdrant | [`backend/`](backend/) | API, Qdrant server và Streamlit client chạy riêng | File, trạng thái tác vụ và vector lưu trên máy chủ |

## 1. Streamlit Cloud

### Chạy ở máy cá nhân

Yêu cầu Python 3.11 trở lên. Chạy các lệnh từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Nhập Gemini API key hoặc OpenAI API key trong sidebar. Người quản trị có thể đặt
`GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong **Streamlit Cloud → Settings → Secrets**
để khách truy cập dùng khóa của ứng dụng. Khi chạy cục bộ, có thể dùng file `.env`
không commit. Gemini là nhà cung cấp mặc định nếu chưa chọn nhà cung cấp khác.
OpenAI cần tài khoản còn hạn mức API; mỗi nhà cung cấp có giới hạn lưu lượng riêng.

Ứng dụng nhận PDF có lớp chữ chọn được và TXT UTF-8, tối đa 20 MiB. Nội dung
được gửi tới nhà cung cấp AI đã chọn để tạo embedding và câu trả lời. PDF scan
cần OCR trước khi tải lên phiên bản Cloud. Chỉ đổi nhà cung cấp sau khi đã xóa
tài liệu trong phiên, vì vector của các mô hình embedding không tương thích.

Mỗi phiên Streamlit giữ tài liệu, vector và lịch sử chat riêng. Dữ liệu có thể
mất khi phiên hoặc ứng dụng khởi động lại. Phiên bản Cloud không gọi FastAPI
trong `backend/` và không phải kho lưu trữ lâu dài cho nhiều người dùng.

## 2. FastAPI + Qdrant

Phiên bản trong [`backend/`](backend/) hỗ trợ lập chỉ mục nền có theo dõi tiến
độ, phân tích bố cục PDF, bảng, hình/biểu đồ bằng Gemini Vision khi được cấu
hình, truy xuất child-to-parent và câu trả lời có trích dẫn vị trí trang PDF.

### Cài đặt

Yêu cầu Python 3.11 trở lên và Qdrant đang chạy. **Chạy các lệnh sau từ thư
mục `backend/`**. Lần lập chỉ mục hoặc hỏi đáp đầu tiên có thể tải trọng số
mô hình local từ Hugging Face.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-backend.txt
cp .env.example .env
docker run -d --name qdrant-genai -p 127.0.0.1:6333:6333 -v qdrant_genai_data:/qdrant/storage qdrant/qdrant
```

Nếu container đã tồn tại, dùng `docker start qdrant-genai`. Sửa `.env` theo
môi trường: `RAG_API_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `RAG_DATA_DIR` và
`MAX_UPLOAD_BYTES`. Streamlit client đọc URL API từ `RAG_API_URL`.

| Cấu hình | Giá trị / tác dụng |
| --- | --- |
| `EMBEDDING_PROVIDER` | `local` (mặc định), `fastembed` hoặc `openai`. Với FastEmbed, cài thêm `pip install fastembed`. |
| `EMBEDDING_BATCH_SIZE` | Số đoạn trong mỗi batch; mặc định 64. |
| `LLM_PROVIDER` | `local` (mặc định), `gemini` hoặc `openai`. |
| `LLM_FALLBACK_PROVIDER` | Nhà cung cấp dự phòng tùy chọn, khác nhà cung cấp chính. |
| `GEMINI_API_KEY` | Cần cho Gemini Vision và khi chọn Gemini làm LLM. Không có khóa, chữ và bảng vẫn được lập chỉ mục; vùng hình có cảnh báo `vision_unconfigured`. |
| `OPENAI_API_KEY` | Cần khi chọn OpenAI cho embedding hoặc LLM. |

Các lệnh gọi dịch vụ AI bên ngoài thử lại lỗi 429/5xx tạm thời với exponential
backoff và jitter. Sau nhiều lỗi liên tiếp, circuit breaker chuyển LLM sang
nhà cung cấp dự phòng trong thời gian cooldown đã cấu hình.

Mở **hai terminal** trong `backend/`, cùng kích hoạt `.venv`:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
python -m streamlit run client.py
```

Mở <http://127.0.0.1:8501> để chat và <http://127.0.0.1:8000/docs> để xem API.
Chỉ chạy một tiến trình FastAPI: tiến trình này quản lý worker lập chỉ mục và
truy vấn, đồng thời giữ các mô hình local trong bộ nhớ.

### Luồng sử dụng

1. Tải PDF hoặc TXT UTF-8 ở sidebar. Giới hạn mặc định là 20 MiB. Phân tích
   hình ảnh, biểu đồ và trang scan cần `GEMINI_API_KEY`.
2. API trả job ID ngay. Giao diện hiển thị tiến độ và vô hiệu hóa ô chat cho
   tới khi tài liệu có trạng thái `completed`.
3. Gửi câu hỏi. Một hành động thực hiện truy xuất và tạo câu trả lời. Câu hỏi
   tóm tắt toàn tài liệu dùng quy trình map-reduce theo các đoạn đã lập chỉ mục.
4. Chọn trích dẫn đánh số để xem trang PDF gốc và vùng chữ được đánh dấu. Với
   TXT, giao diện hiển thị đoạn trích dẫn.

Lịch sử chat nằm trong phiên Streamlit và được tách theo tài liệu. Trong phần
cài đặt nâng cao, Top-K mặc định là 3, ngưỡng cosine mặc định là 0.55. Điểm
tương đồng vector không phải độ tin cậy của câu trả lời.

### Xử lý và lưu trữ

- `backend/upload.py` ghi file upload theo luồng vào `uploads/`; kiểm tra định
  dạng, kích thước, phần đầu PDF và mã hóa UTF-8.
- `backend/rag/extractor.py` kết hợp tọa độ chữ/bảng của pdfplumber với vùng
  ảnh và vector của PyMuPDF. Bảng được chuyển thành Markdown; Vision tạo mô tả
  định lượng và bảng dữ liệu cho hình. Lỗi đọc từng trang được ghi cảnh báo để
  các trang còn lại tiếp tục xử lý.
- `backend/rag/transform.py` giữ nguyên mỗi bảng, khối code và hình trong một
  parent. Các child chunk theo câu và vector theo hàng bảng trỏ về parent đó.
- `backend/rag/embedding.py` hỗ trợ MiniLM local, FastEmbed tùy chọn và OpenAI
  theo batch. Lệnh gọi dịch vụ ngoài có retry giới hạn.
- `backend/rag/repository.py` lưu point ID ổn định trong Qdrant cosine. Vector
  child giữ ngữ cảnh parent và tọa độ; retrieval loại trùng parent trước khi
  đưa ngữ cảnh vào LLM.
- `backend/rag/state.py` lưu tài liệu và job vào `var/jobs.sqlite3` hoặc thư
  mục cấu hình. Job có trạng thái `queued`, `processing`, `completed`, `failed`,
  cùng tiến độ, bước xử lý, kết quả và lỗi. Job gián đoạn được xếp lại khi khởi
  động.
- `backend/rag/pipeline.py` truy xuất một lần cho câu hỏi thông thường. Câu
  hỏi tóm tắt duyệt toàn bộ nội dung rồi tổng hợp theo từng phần.
- `backend/rag/llm.py` giới hạn prompt theo các parent hoàn chỉnh và hỗ trợ
  local Qwen, Gemini, OpenAI cùng cơ chế dự phòng.

Tên collection được tạo từ `QDRANT_COLLECTION` với hậu tố
`_parent_v1_{provider}`. Collection cũ vẫn được giữ; tài liệu thuộc phiên bản
index cũ sẽ được xếp hàng lập chỉ mục lại khi khởi động.

### API

| Endpoint | Chức năng |
| --- | --- |
| `POST /upload` | Lưu file, xếp hàng lập chỉ mục; trả HTTP 202 và job ID. |
| `GET /documents`, `GET /documents/{source}` | Xem trạng thái và số lượng nội dung của tài liệu. |
| `GET /documents/{source}/file` | Đọc file gốc cho trích dẫn PDF. |
| `GET /documents/{source}/figures/{figure_name}` | Đọc ảnh đã tách từ tài liệu. |
| `POST /documents/{source}/index` | Xếp hàng lập chỉ mục lại. |
| `POST /chat` | Xếp hàng truy xuất và tạo câu trả lời. |
| `POST /search` | Chỉ xếp hàng truy xuất. |
| `GET /jobs`, `GET /jobs/{job_id}` | Xem tiến độ và kết quả job. |

Khi tài liệu đang được lập chỉ mục, API từ chối tạo chat job của tài liệu đó
với HTTP 409; giao diện cũng khóa ô chat. Lỗi API có dạng
`{code, message, suggested_action}`. Job lập chỉ mục có thể kèm `warnings` cho
trang hoặc hình không đọc được.

Backend chưa có xác thực. Giữ nó trên localhost hoặc bổ sung xác thực và lưu
trữ phù hợp trước khi mở ra mạng. Vùng đánh dấu PDF dựa trên tọa độ chữ có thể
rộng hơn mong muốn với tài liệu bố cục phức tạp. Không commit API key, file
`.env`, tài liệu riêng tư hoặc dữ liệu trong `uploads/` và `var/` lên GitHub.
