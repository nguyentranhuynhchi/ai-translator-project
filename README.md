# 🌐 Benchmarking Machine Translation Ecosystems: From Sequential Architectures to Edge-Optimized Multi-Adapter LLMs

Dự án là một nghiên cứu thực nghiệm toàn diện về các giải pháp Dịch máy song phương Anh ⇄ Việt, trải dài từ các kiến trúc mạng tuần tự cổ điển (RNN, LSTM, GRU, BiLSTM), qua mô hình Attention-based (Scratch Transformer), đến giải pháp tối ưu hóa biên (Edge AI) bằng kỹ thuật **4-bit QLoRA Multi-Adapter** trên nền Đại mô hình ngôn ngữ `Qwen2.5-0.5B`.

Mục tiêu của dự án là giải quyết một bài toán thực tế trong môi trường doanh nghiệp: **bảo mật dữ liệu tuyệt đối (Zero-Data-Leakage)** và **tối ưu hóa tài nguyên phần cứng cực hạn (~1GB VRAM)**, cho phép vận hành hệ thống offline hoàn toàn trên thiết bị Edge hoặc máy tính cá nhân phổ thông.

---

## 🏗️ 1. Kiến trúc Hệ thống & Luồng xử lý Dữ liệu

Hệ thống được thiết kế theo mô hình **"Một Xác Nhiều Hồn" (Single Backbone, Multi-Soul)**: mô hình nền được đóng băng và cô lập, trong khi các "bộ não ngôn ngữ" (adapter) được định tuyến động theo thời gian thực dựa trên lựa chọn của người dùng.

### Luồng tương tác

<img width="1920" height="617" alt="{48C45EA0-71F5-4AC6-BAD4-EF976D656AAD}" src="https://github.com/user-attachments/assets/b13e0b50-94a2-4928-a83a-27d7e57d4dc6" />
<img width="1920" height="739" alt="{C9A0ED77-C8F3-46B0-A678-6FB77772B611}" src="https://github.com/user-attachments/assets/e2ba99b9-eb44-4526-96c8-d51920ce8422" />
<img width="1920" height="422" alt="{764E3491-9D30-4BE6-AF59-C461F37F72D9}" src="https://github.com/user-attachments/assets/6c6795f2-c215-47f0-900e-b2afcd8e86ed" />


### Các bước vận hành

1. **Khởi tạo local:** toàn bộ trọng số mô hình nền (`base_qwen` — `Qwen2.5-0.5B`), tokenizer, và cấu hình adapter được tải trực tiếp từ ổ cứng lên VRAM, không cần kết nối mạng ngoài.
2. **Định tuyến prompt:** dựa vào chiều dịch đã chọn, hệ thống tự động bọc văn bản vào khung prompt tương ứng, định nghĩa tại `constants.py`.
3. **Chuyển mạch adapter động:** cơ chế `.set_adapter()` hoán đổi các ma trận trọng số LoRA tương ứng trong thời gian thực (≤ 1ms) mà không cần reload cấu trúc mô hình nền.

---

## 📊 2. Nhật ký Thực nghiệm & Kết quả Đo đạc

Mọi cấu hình được đánh giá bằng hai thang đo chuẩn quốc tế: **SacreBLEU** (↑, độ tương đồng phân phối n-gram) và **TER** (↓, Translation Error Rate — tỷ lệ lỗi chỉnh sửa).

**Hạ tầng đánh giá:**
- Mạng tuần tự & Transformer từ đầu: huấn luyện trên tập ngữ liệu song ngữ PhoMT.
- LLM (QLoRA): tất cả các phiên bản được đánh giá độc lập trên cùng một tập test cố định gồm **1.500 câu**, đảm bảo tính nhất quán so sánh.

### Bảng 1 — Kiến trúc tuần tự (Sequential Models)
*Cấu hình: Embedding Dim = 256, Hidden Dim = 512, Vocab Size = 10.000 (Joint BPE qua SentencePiece)*

| Kiến trúc | Giải mã | SacreBLEU ↑ | TER ↓ | Nhận xét |
|---|---|:---:|:---:|---|
| **Vanilla RNN** | Beam Search (K=3) | 0.45 | 120.42 | Mất mát thông tin nghiêm trọng với câu dài do vanishing gradient. |
| | Argmax (Greedy) | 0.28 | 172.44 | Nghẽn bộ nhớ khiến dịch lặp từ vô nghĩa. |
| **LSTM** | Beam Search (K=3) | 2.26 | 106.99 | Khá hơn RNN nhờ cơ chế cổng (gates), bắt đầu giữ được cấu trúc ngữ pháp ngắn. |
| | Argmax (Greedy) | 1.99 | 111.48 | Vẫn dịch thiếu/sót từ ở câu có sắc thái phức tạp. |
| **BiLSTM** | Beam Search (K=3) | 4.15 | 106.75 | Cải thiện nhờ quét ngữ cảnh hai chiều, tăng kết nối ngữ nghĩa. |
| | Argmax (Greedy) | 3.49 | 114.62 | Vẫn bị giới hạn bởi context vector bottleneck. |
| **GRU** | Beam Search (K=3) | **4.34** | **103.76** | Kiến trúc tối ưu nhất nhóm tuần tự — ít tham số hơn LSTM, hội tụ nhanh, BLEU cao nhất nhóm. |
| | Argmax (Greedy) | 3.74 | 109.41 | |

### Bảng 2 — Attention-based (tự huấn luyện từ đầu)
*Cấu hình: Vanilla Seq2Seq Transformer chuẩn, làm mốc so sánh*

| Kiến trúc | Giải mã | SacreBLEU ↑ | TER ↓ | Nhận xét |
|---|---|:---:|:---:|---|
| **Scratch Transformer** | Beam Search (K=3) | **10.49** | **76.29** | Bước nhảy vọt về chất — Multi-Head Self-Attention loại bỏ tính tuần tự, liên kết tốt các token phân tách xa. BLEU gấp ~2.5 lần GRU. |
| | Argmax (Greedy) | 10.39 | 77.08 | |

### Bảng 3 — LLM + QLoRA Multi-Adapter
*Cấu hình nền: Qwen2.5-0.5B, đóng băng ở 4-bit (NF4). Đánh giá offline trên tập test cố định 1.500 câu.*

| Cấu hình LoRA (Rank/Alpha) | Dữ liệu train | Chiều dịch | Siêu tham số | SacreBLEU ↑ | TER ↓ | Ghi chú |
|---|---|:---:|---|:---:|:---:|---|
| **Gốc (Zero-shot)** | không train | En→Vi | N/A | 5.89 | 88.33 | Sinh từ mất kiểm soát do chưa được align với tập song ngữ đặc thù. |
| | | Vi→En | | 14.91 | 75.09 | |
| **LoRA r=16, α=32** | 800 câu | En→Vi | Step=400, BS=2 | 14.50 | 71.10 | Chỉ với 800 câu (quick demo), việc cập nhật trọng số ma trận chiếu attention đã giúp vượt qua Scratch Transformer. |
| | | Vi→En | | 14.48 | 78.86 | |
| **LoRA r=8, α=16** | 800 câu | En→Vi | Step=400, BS=2 | 13.50 | 77.05 | Hạ rank xuống 8 thu hẹp không gian tham số, gây suy giảm nhẹ BLEU. |
| | | Vi→En | | 14.02 | 71.75 | |
| **LoRA r=8, α=16 (lần 2)** | 5.000 câu | En→Vi | Epoch=3, BS=4, Accum=2 | 16.22 | 69.43 | Tăng data lên 5k câu cải thiện rõ rệt — data lớn bù đắp được giới hạn của rank thấp. |
| | | Vi→En | | 15.47 | 79.18 | |
| **LoRA r=32, α=64** | 15.000 câu / epoch | En→Vi | Epoch=3, BS=1, Accum=4 | **18.41** | **66.88** | **Cấu hình tối ưu nhất** — rank 32 kết hợp data lớn giúp mô hình đạt độ chín văn phong, dịch tự nhiên, chuẩn bản địa. |
| **LoRA r=32, α=64 (lần 2)** | 15.000 câu / epoch | Vi→En | Epoch=3, BS=1, Accum=4 | **16.88** | **71.87** | Làm chủ tốt cấu trúc đảo ngữ phức tạp từ Việt sang Anh. |

---

## 🧠 3. Lập luận Kỹ thuật

1. **Giới hạn của mạng recurrent:** thực nghiệm cho thấy mạng tuần tự truyền thống khó áp dụng vào sản phẩm dịch thuật thương mại do hiện tượng thắt nút thông tin — nén cả câu dài vào một vector kích thước cố định làm mất tính phân cấp ngôn ngữ.

2. **Tính kinh tế của QLoRA Multi-Adapter trong doanh nghiệp:** thay vì full fine-tuning toàn bộ mô hình (tốn kém và gây catastrophic forgetting), giải pháp gắn các adapter LoRA siêu nhẹ lên base model 4-bit (NF4) mang lại:
   - **Chi phí phần cứng gần như bằng 0:** tổng VRAM tiêu thụ thực tế chỉ khoảng **~1GB** (0.25GB base Qwen2.5-0.5B + 0.05GB adapter + 0.7GB buffer ngữ cảnh & CUDA kernels). Chạy mượt trên laptop văn phòng hoặc thiết bị Edge AI thông thường.
   - **Bảo mật tuyệt đối (Zero-Data-Leakage):** chạy offline 100% tại local, loại bỏ hoàn toàn rủi ro rò rỉ dữ liệu ra internet — tiêu chuẩn bắt buộc với các tổ chức tài chính, y tế, pháp lý khi xử lý văn bản nội bộ.

---

## 🛠️ 4. Tech Stack & Hướng dẫn Triển khai

### Core Technologies
- **Deep Learning Framework:** `torch` (CUDA-accelerated)
- **LLM Engine:** `transformers`, `bitsandbytes` (4-bit quantization NF4)
- **Parameter-Efficient Fine-Tuning:** `peft` (LoRA)
- **Production UI:** `gradio` (Modern Fluent CSS Theme)

### Chạy local (CMD)

> Môi trường ảo Python được thiết lập cô lập bên trong thư mục Conda vật lý của máy.

```cmd
:: 1. Di chuyển vào thư mục dự án
D:
cd D:\DA_CaNhan\ai-translator-project

:: 2. Khởi chạy ứng dụng bằng Python của môi trường ảo local
D:\C\anaconda3\envs\ai_gradio\python.exe app.py
```

Sau khi terminal thông báo khởi chạy thành công, truy cập **http://127.0.0.1:7860** trên trình duyệt để trải nghiệm giao diện.
