# Báo cáo Lab 6 Part 1 – Giới thiệu về Transformers

## Bài 1 – Masked Language Modeling (BERT)

**Câu 1. Mô hình đã dự đoán đúng từ "capital" không?**  
Có. Kết quả trong `fill_mask.txt` cho thấy dự đoán top-1 là `' capital'` với xác suất khoảng 0.93, tạo thành câu *"Hanoi is the capital of Vietnam."*, hoàn toàn đúng.

**Câu 2. Tại sao các mô hình encoder-only như BERT phù hợp cho tác vụ này?**  
- BERT được pre-train đúng bằng bài toán *Masked Language Modeling*, nên tối ưu cho việc đoán token bị che.  
- Kiến trúc bidirectional encoder cho phép dùng cả ngữ cảnh bên trái (*"Hanoi is the"*) và bên phải (*"of Vietnam"*) khi dự đoán token `<mask>`.  
- Encoder-only tập trung vào biểu diễn ngữ nghĩa của câu, rất phù hợp cho nhiệm vụ "điền vào chỗ trống" dựa trên ngữ cảnh toàn câu.

---

## Bài 2 – Text Generation (GPT)

**Câu 1. Kết quả sinh ra có hợp lý không?**  
Có. Đoạn văn trong `text_generation.txt` nói về việc học NLP, lợi ích, các vấn đề có thể giải quyết,... đúng chủ đề bắt đầu từ câu mồi *"The best thing about learning NLP is"*, câu cú nhìn chung mạch lạc dù có lặp ý.

**Câu 2. Tại sao các mô hình decoder-only như GPT phù hợp cho tác vụ này?**  
- GPT được pre-train với mục tiêu dự đoán từ tiếp theo (next token), trùng với bài toán sinh văn bản.  
- Kiến trúc decoder-only, unidirectional chỉ nhìn các token phía trước, rất tự nhiên cho việc sinh chuỗi từ trái sang phải.  
- Mô hình hóa tốt phân phối P(token_t | token_1…token_{t-1}), nên có thể sinh các đoạn văn dài tương đối mạch lạc.

---

## Bài 3 – Vector biểu diễn câu (Sentence Representation với BERT)

**Câu 1. Kích thước vector biểu diễn là bao nhiêu? Tương ứng với tham số nào của BERT?**  
Kết quả trong `sentence_embedding.txt` cho thấy vector câu có kích thước **(1, 768)**.  
Con số **768** chính là tham số **`hidden_size`** của mô hình `bert-base-uncased` (kích thước vector ẩn cho mỗi token, cũng là kích thước vector câu khi Mean Pooling).

**Câu 2. Tại sao cần dùng `attention_mask` khi Mean Pooling?**  
- Khi padding, có thêm token đệm không mang thông tin. Nếu tính trung bình cả các vị trí padding thì vector câu sẽ bị pha loãng, sai lệch.  
- `attention_mask` cho biết vị trí token thật (1) và padding (0). Ta chỉ cộng và chia trung bình trên các vị trí có mask = 1, nhờ đó vector câu phản ánh đúng nội dung câu, không bị ảnh hưởng bởi padding.