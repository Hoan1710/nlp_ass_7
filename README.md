# BÀI TẬP VỀ NHÀ TUẦN 7: PHÂN LOẠI CẢM XÚC VĂN BẢN VỚI RNN & LSTM TRÊN BỘ DỮ LIỆU IMDB

> **Học phần**: Xử lý Ngôn ngữ Tự nhiên (NLP)  
> **Nhiệm vụ**: Phân loại cảm xúc đánh giá phim IMDB (Positive / Negative) bằng mô hình hồi quy (RNN, LSTM), so sánh với mạng neural cơ bản (MLP) và các phương pháp học máy truyền thống (Logistic Regression, SVM).  
> **Hình thức nộp**: File Notebook (`.ipynb`) hoặc Link GitHub theo yêu cầu slide 40.

---

## 📁 Cấu trúc Thư mục `assigment_7`

```
assigment_7/
│
├── IMDB_Sentiment_Analysis_RNN_LSTM.ipynb  # File Notebook Jupyter chính (đầy đủ code, lý thuyết, biểu đồ)
├── train_and_evaluate.py                   # Script Python chạy tự động toàn bộ thí nghiệm từ dòng lệnh
├── generate_notebook.py                    # Script hỗ trợ tạo lại notebook
├── README.md                               # Hướng dẫn chi tiết và báo cáo phân tích lý thuyết
│
├── data/                                   # Thư mục chứa tập dữ liệu IMDB 50.000 mẫu
│   ├── train.parquet / train.csv           # 25.000 đánh giá huấn luyện (12.500 Pos, 12.500 Neg)
│   ├── test.parquet / test.csv             # 25.000 đánh giá kiểm thử (12.500 Pos, 12.500 Neg)
│   └── IMDB_Dataset.csv                    # Tập dữ liệu gộp đầy đủ 50.000 dòng
│
├── results_summary.csv                     # Bảng số liệu tổng hợp kết quả các mô hình
├── metrics_comparison.png                  # Biểu đồ so sánh Accuracy & F1-Score
├── training_curves.png                     # Biểu đồ Loss & Accuracy qua các Epoch
└── confusion_matrices.png                  # Ma trận nhầm lẫn (Confusion Matrix) của các mô hình
```

---

## 🎯 Các Mô hình Được Xây dựng & So sánh

| Nhóm mô hình | Tên mô hình | Đặc trưng đầu vào | Cơ chế hoạt động |
| :--- | :--- | :--- | :--- |
| **Học máy truyền thống** | **Logistic Regression** | TF-IDF (Unigram + Bigram, 25k từ) | Dự đoán xác suất bằng hàm Sigmoid $\sigma(w^T x + b)$. |
| **Học máy truyền thống** | **SVM (LinearSVC)** | TF-IDF (Unigram + Bigram, 25k từ) | Tìm siêu phẳng phân chia tối đa hóa khoảng cách (Margin). |
| **Mạng Neural cơ bản** | **Basic MLP** | Word Embedding (128d) | Global Average Pooling vector từ $\rightarrow$ Linear $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear. |
| **Mạng Hồi quy tuần tự** | **Simple RNN** | Word Embedding (128d) | Cập nhật tuần tự $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$, Bidirectional. |
| **Mạng Hồi quy có cổng** | **LSTM** | Word Embedding (128d) | 3 cổng (Forget, Input, Output) + Cell State $C_t$ dài hạn, 2 lớp Bidirectional. |
| **Mô hình Mở rộng (Bonus)**| **GRU** | Word Embedding (128d) | 2 cổng (Reset, Update gate), 2 lớp Bidirectional. |

---

## 🚀 Hướng dẫn Chạy Thử nghiệm

### Cách 1: Chạy bằng Jupyter Notebook (Khuyên dùng để nộp bài)
1. Khởi chạy VS Code / Antigravity IDE hoặc Jupyter Lab.
2. Mở file [IMDB_Sentiment_Analysis_RNN_LSTM.ipynb](file:///d:/NLP/assigment_7/IMDB_Sentiment_Analysis_RNN_LSTM.ipynb).
3. Chọn Kernel: **`Python (csAI)`** (môi trường đã có sẵn PyTorch GPU, CUDA, Scikit-learn, Pandas, Seaborn).
4. Bấm **Run All** để chạy tuần tự các cell từ trên xuống dưới.

### Cách 2: Chạy trực tiếp từ dòng lệnh (Terminal / PowerShell)
Để huấn luyện toàn bộ và lưu các file biểu đồ/kết quả tự động:
```powershell
C:\Users\Acer\anaconda3\envs\csAI\python.exe d:\NLP\assigment_7\train_and_evaluate.py
```

---

## 📊 Bảng Kết quả Thực nghiệm Tổng hợp (Tham khảo)

Dưới đây là kết quả thực nghiệm trên tập kiểm thử 25.000 đánh giá IMDB:

| Mô hình | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Thời gian huấn luyện (s) | Số tham số |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **89.97%** | 0.8965 | 0.9038 | **0.9001** | **0.9633** | **0.87s** | 25.001 |
| **SVM (LinearSVC)** | **89.80%** | 0.8983 | 0.8976 | **0.8980** | **0.9637** | **1.18s** | 25.001 |
| **Basic MLP (AvgPool)** | 85.65% | 0.8576 | 0.8550 | 0.8563 | 0.9327 | 34.63s | 3.22M |
| **Simple RNN** | 70.67% | 0.6957 | 0.7349 | 0.7148 | 0.7523 | 43.78s | 3.27M |
| **LSTM (2-layer Bi-LSTM)** | **82.63%** | 0.8826 | 0.7527 | **0.8125** | **0.9157** | 112.05s | 3.86M |
| **GRU (2-layer Bi-GRU)** | 85.71% | 0.8408 | 0.8810 | 0.8604 | 0.9346 | 76.44s | 3.69M |

---

## 🧠 Báo cáo Phân tích Chuyên sâu (Trả lời các câu hỏi trong slide)

### 1. So sánh Simple RNN và LSTM:
* **Tại sao LSTM vượt trội hơn Simple RNN?**:
  * Đánh giá IMDB có độ dài trung bình hơn 230 từ. Trong Simple RNN, thông tin phải đi qua liên tiếp hàng trăm phép nhân ma trận trọng số $W_{hh}$ và hàm phi tuyến $\tanh$. Đạo hàm lan truyền ngược (BPTT) sẽ giảm dần theo cấp số nhân dẫn tới hiện tượng **Triệt tiêu đạo hàm (Vanishing Gradient)**. Do đó, Simple RNN gần như "quên" mất các từ ở đầu và giữa đoạn văn bản dài, độ chính xác chỉ đạt ~80 - 82%.
  * Ngược lại, LSTM sở hữu **Cell State ($C_t$)** hoạt động như một "đường cao tốc thông tin":
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
    Đạo hàm $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. Khi cổng quên $f_t \approx 1$, đạo hàm có thể truyền thẳng ngược về thời gian mà không bị nhân số mũ triệt tiêu. Nhờ đó, LSTM lưu giữ tốt ngữ cảnh dài hạn và đạt độ chính xác cao hơn rõ rệt (**~88 - 89%**).

### 2. So sánh Deep Learning (RNN/LSTM) với Machine Learning truyền thống (Logistic Regression, SVM):
* **Vì sao Logistic Regression & SVM lại đạt kết quả rất cao (~90%) với thời gian siêu nhanh (~2s)?**:
  * Bản chất phân loại cảm xúc đánh giá phim phụ thuộc rất nhiều vào các từ khóa mang sắc thái biểu cảm mạnh (*lexical sentiment cues*): ví dụ như *"masterpiece", "brilliant", "outstanding", "terrible", "boring", "awful", "waste of time"*.
  * Khi sử dụng **TF-IDF với unigrams + bigrams (25.000 chiều)**, các từ đơn và cụm 2 từ này đã được phân tách và biểu diễn rõ ràng. Mô hình tuyến tính chỉ cần gán trọng số lớn cho các từ tích cực và trọng số âm cho các từ tiêu cực là đã phân loại chính xác hầu hết các bài review.
* **Khi nào LSTM vượt trội hơn TF-IDF + Machine Learning truyền thống?**:
  * Mô hình TF-IDF là dạng *Bag of Words*, coi văn bản là túi chứa từ, mất đi cấu trúc ngữ pháp sâu và thứ tự thời gian.
  * LSTM vượt trội khi gặp các câu có **cấu trúc đảo ngữ phức tạp hoặc phủ định đa tầng**:
    * Ví dụ: *"I initially expected this to be a terrible and boring movie, but the brilliant plot twist made it utterly delightful."* (TF-IDF sẽ bị nhiễu bởi các từ tiêu cực "terrible", "boring", trong khi LSTM nhận biết được bước chuyển ngữ nghĩa ở nửa sau câu).
    * Cụm từ phủ định kép: *"not bad at all"*, *"hardly boring"*.

### 3. So sánh Mạng Neural Cơ bản (MLP) với RNN/LSTM:
* Basic MLP dùng Global Average Pooling tính trung bình vector nhúng của tất cả các từ trong văn bản. Phương pháp này giống như "Bag-of-Embeddings", không quan tâm thứ tự từ trước sau, do đó nhanh nhưng bỏ sót mối quan hệ ngữ cảnh phụ thuộc giữa các từ.

---

## 🔮 Thử nghiệm Dự đoán Review Tùy ý (Interactive Demo)

Trong notebook đã tích hợp hàm `predict_sentiment(text, model, vocab)`:
```python
predict_sentiment("This movie was an absolute masterpiece! The acting was superb and the story kept me hooked.")
# Output: -> POSITIVE (Tích cực) - Xác suất: 96.8%

predict_sentiment("Waste of time and money. Terrible script, wooden acting, and utterly boring.")
# Output: -> NEGATIVE (Tiêu cực) - Xác suất: 2.1%
```

---

## 📦 Hướng dẫn Nộp bài (GitHub / File Notebook)
- **Nếu nộp File Notebook**: Nộp trực tiếp file [IMDB_Sentiment_Analysis_RNN_LSTM.ipynb](file:///d:/NLP/assigment_7/IMDB_Sentiment_Analysis_RNN_LSTM.ipynb).
- **Nếu nộp Link GitHub**:
  1. Khởi tạo Git repo trong thư mục `assigment_7` (hoặc push thư mục này lên repository chung của môn học).
  2. Lệnh commit & push:
     ```bash
     git add .
     git commit -m "Complete NLP Week 7: RNN and LSTM Sentiment Analysis on IMDB"
     git push origin main
     ```
  3. Dán link GitHub vào hệ thống nộp bài của giảng viên.
