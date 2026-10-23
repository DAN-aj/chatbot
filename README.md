# Talk to Documents

Ứng dụng hỏi đáp tài liệu PDF/TXT bằng RAG. Repository có hai cách chạy độc lập:

| Phiên bản | Mã nguồn | Cách chạy | Lưu trữ |
| --- | --- | --- | --- |
| [Streamlit Cloud](https://chatbot-ni241maz32f.streamlit.app/) | `streamlit_app.py`, `cloud_rag.py`, `providers.py` | Một ứng dụng Streamlit; không cần FastAPI | Qdrant local và tài liệu trong phiên Streamlit |
| FastAPI + Qdrant | [`backend/`](backend/README.md) | API, Qdrant server và Streamlit client chạy riêng | File, trạng thái tác vụ và vector lưu bền trên máy chủ |

## Chạy ứng dụng Streamlit Cloud ở máy cá nhân

Yêu cầu Python 3.11 trở lên:

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

Ứng dụng nhận PDF có lớp chữ chọn được và TXT UTF-8, tối đa 20 MiB. Nội dung được
gửi tới nhà cung cấp AI đã chọn để tạo embedding và câu trả lời. Với PDF scan,
hãy OCR trước khi tải lên phiên bản Cloud. Chỉ đổi nhà cung cấp sau khi đã xóa
tài liệu trong phiên, vì vector của hai mô hình embedding không tương thích.

Mỗi phiên Streamlit giữ tài liệu, vector và lịch sử chat riêng. Dữ liệu có thể
mất khi phiên hoặc ứng dụng khởi động lại. Đây không phải kho lưu trữ lâu dài
cho nhiều người dùng. Không commit khóa API hoặc tài liệu riêng tư lên GitHub.

## FastAPI + Qdrant

Phiên bản [`backend/`](backend/README.md) hỗ trợ lập chỉ mục nền có theo dõi
tiến độ, phân tích bố cục PDF, bảng, hình/biểu đồ bằng Vision khi cấu hình
Gemini, truy xuất child-to-parent và câu trả lời có trích dẫn vị trí trang PDF.
Nó chạy độc lập với ứng dụng Streamlit Cloud ở thư mục gốc. Xem hướng dẫn cài
đặt, biến môi trường và API trong [README backend](backend/README.md).
