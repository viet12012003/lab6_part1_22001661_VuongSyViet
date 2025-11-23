from transformers import pipeline
from pathlib import Path

# 1. Tải pipeline "fill-mask"
# Pipeline này sẽ tự động tải một mô hình mặc định phù hợp (thường là một biến thể của BERT)
mask_filler = pipeline("fill-mask")

# 2. Câu đầu vào với token [MASK]
input_sentence = "Hanoi is the <mask> of Vietnam."

# 3. Thực hiện dự đoán
# top_k=5 yêu cầu mô hình trả về 5 dự đoán hàng đầu
predictions = mask_filler(input_sentence, top_k=5)

# 4. In kết quả
print(f"Câu gốc: {input_sentence}")
for pred in predictions:
    print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    print(f" -> Câu hoàn chỉnh: {pred['sequence']}")

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)
lines = [f"Câu gốc: {input_sentence}"]
for pred in predictions:
    lines.append(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    lines.append(f" -> Câu hoàn chỉnh: {pred['sequence']}")
out_file = results_dir / "fill_mask.txt"
with out_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))
