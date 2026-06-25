# Báo cáo Fine-tuning Qwen2.5-3B với LoRA/QLoRA

## 1. Tổng quan cấu hình (Profile: T4)
- **Model gốc:** `unsloth/Qwen2.5-3B-bnb-4bit`
- **Thiết bị:** Tesla T4 (15.6 GB VRAM)
- **Dataset:** 5CD-AI/Vietnamese-alpaca (200 samples)
- **Max Sequence Length:** 1024
- **Phương pháp:** QLoRA (4-bit quantization + LoRA adapters)

## 2. Kết quả thực nghiệm Rank (Rank Experiment)
Chúng ta đã tiến hành train 3 phiên bản LoRA adapter với các rank khác nhau để so sánh hiệu năng:

|   rank |   alpha |   trainable_params |   train_time_min |   peak_vram_gb |   eval_loss |   eval_perplexity |
|-------:|--------:|-------------------:|-----------------:|---------------:|------------:|------------------:|
|      8 |      16 |        1.8432e+06  |          3.91681 |        7.21614 |     1.55769 |           4.74786 |
|     16 |      32 |        3.6864e+06  |          4.1446  |        6.61775 |     1.51608 |           4.55435 |
|     64 |     128 |        1.47456e+07 |          3.86239 |        7.99851 |     1.47681 |           4.37898 |

### Nhận xét:
- **Trainable Params:** Rank 64 sử dụng gấp 8 lần tham số so với Rank 8 nhưng vẫn nằm trong ngưỡng an toàn của T4 (~14.7M params).
- **VRAM:** Peak VRAM dao động từ 6.6GB đến 8.0GB, chứng tỏ cấu hình này cực kỳ tối ưu cho GPU 16GB.
- **Loss/Perplexity:** Rank 64 cho kết quả tốt nhất (Perplexity thấp nhất: 4.3790), cho thấy việc tăng rank giúp mô hình học được nhiều đặc trưng hơn từ dữ liệu tiếng Việt.

## 3. Đánh giá định tính (Qualitative Comparison)
So sánh phản hồi giữa Base Model và Fine-tuned Model (phiên bản r=16):


### Ví dụ 1: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base Model:** Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác ...
- **Fine-tuned:** Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc...

### Ví dụ 2: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base Model:** Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:

```python
def fibonacci(n):
    if n <= 0:
        return "N phải là một số dương"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
 ...
- **Fine-tuned:** Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:

```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range...

### Ví dụ 3: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base Model:** 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru...
- **Fine-tuned:** 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với ...

### Ví dụ 4: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base Model:** LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ...
- **Fine-tuned:** LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ...

### Ví dụ 5: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base Model:** Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu...
- **Fine-tuned:** Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th...


## 4. Chi phí & Thời gian
- **Tổng thời gian training (3 ranks):** 11.9 phút
- **Chi phí ước tính:** $0.07 (Dựa trên đơn giá $0.35/giờ)

## 5. Kết luận
1. **Hiệu quả của Rank:** Rank cao (r=64) mang lại độ chính xác (perplexity) tốt hơn mà không làm tăng đáng kể thời gian training trên tập dữ liệu nhỏ.
2. **Khả năng đáp ứng:** Sau khi fine-tune, mô hình có xu hướng trả lời bằng tiếng Việt tự nhiên hơn và bám sát cấu trúc Alpaca.
3. **Khuyến nghị:** Đối với các tác vụ phức tạp hơn, nên giữ rank ở mức 32 hoặc 64 để đạt được sự cân bằng tốt nhất.
