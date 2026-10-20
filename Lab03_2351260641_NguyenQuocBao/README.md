# Lab 3 - Nhận dạng tiếng nói bằng Gaussian HMM

**Học phần:** CSE457 - Xử lý âm thanh và tiếng nói  
**Sinh viên:** Nguyễn Quốc Bảo  
**Mã sinh viên:** 2351260641

## 1. Giới thiệu

Dự án xây dựng bộ nhận dạng 5 lệnh tiếng Việt: `mo`, `dong`, `len`, `xuong`, `dung` (mở, đóng, lên, xuống, dừng). Mỗi lệnh có một Gaussian HMM riêng. Âm thanh được chuyển thành đặc trưng MFCC + Delta + Delta-Delta 39 chiều, sau đó HMM chấm điểm log-likelihood để chọn nhãn có điểm cao nhất.

Dữ liệu trong lần chạy này gồm 40 file WAV tự thu âm: 30 file train (6 file/nhãn) và 10 file test (2 file/nhãn). Tập test nhỏ, nên kết quả chỉ nên xem là thực nghiệm ban đầu.

## 2. Môi trường

Các phiên bản đã được kiểm tra trong môi trường `.venv`:

- Python 3.12.x
- NumPy 2.5.4
- SciPy 1.18.1
- librosa 1.0.0
- scikit-learn 1.9.1
- hmmlearn 0.3.3

Đây là phiên bản của môi trường đã chạy thực tế; phiên bản có thể khác trên máy khác.

### Cài đặt trên Windows (CMD)

Mở Terminal tại thư mục dự án:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn joblib sounddevice
```

Nếu đã có `.venv`, không cần tạo lại môi trường. Nếu PowerShell không chạy được lệnh activate, có thể dùng CMD trong VS Code hoặc tham khảo cú pháp kích hoạt riêng của PowerShell.

## 3. Cấu trúc dữ liệu

```text
dataset/
  train/
    mo/       # 6 WAV
    dong/     # 6 WAV
    len/      # 6 WAV
    xuong/    # 6 WAV
    dung/     # 6 WAV
  test/
    mo/       # 2 WAV
    dong/     # 2 WAV
    len/      # 2 WAV
    xuong/    # 2 WAV
    dung/     # 2 WAV
features/
  train/ ...  # ma trận đặc trưng .npy
  test/  ...
models/
  hmm_models.joblib
results/       # biểu đồ và bảng kết quả
```

Không đặt cùng một file trong cả train và test. WAV được tạo bởi `record_dataset.py` với mono, 16 kHz, PCM 16-bit; mỗi lượt thu âm dài khoảng 2 giây và chèn khoảng im lặng ngắn ở đầu/cuối. Hãy nghe lại các file để xác nhận giọng nói rõ ràng.

## 4. Thứ tự chạy chương trình

Chạy các lệnh từ thư mục gốc dự án, trong môi trường `.venv`:

| Thứ tự | Lệnh | Mục đích / đầu ra |
|---|---|---|
| 1 | `python prepare_dataset.py` | Tạo thư mục `dataset/train` và `dataset/test`. |
| 2 | `python record_dataset.py` | Thu âm dữ liệu; chạy khi cần ghi âm mới. Nếu đã có đủ 40 file, không cần ghi lại. |
| 3 | `python step4_extract_features.py` | Trích 13 MFCC, Delta, Delta-Delta; lưu `.npy` trong `features/`; tạo `results/feature_heatmaps.png`. |
| 4 | `python step5_train_hmm.py` | Huấn luyện 5 HMM và lưu `models/hmm_models.joblib`. |
| 5 | `python step6_evaluate.py` | Đánh giá HMM trên tập test; tạo likelihood CSV, confusion matrix và 2 hình Viterbi. |
| 6 | `python step7_compare_dtw.py` | So sánh HMM với baseline DTW; lưu bảng so sánh và confusion matrices. |
| 7 | `python step8_lexicon.py` | Đọc `lexicon.txt` và kiểm tra parser. |
| 8 | `python step9_bigram.py` | Đếm unigram/bigram từ `corpus_bigram.txt`, tính add-k smoothing. |
| 9 | `python step9_rerank.py` | Minh họa kết hợp acoustic score và bigram score với lambda LM. |
| 10 | `python step10_robustness.py` | Thêm nhiễu Gaussian mức SNR 15 dB vào test và đánh giá HMM. |
| 11 | `python step11_states.py` | So sánh 3, 5, 7 states; lưu `results/state_count_results.csv`. |
| 12 | `python step12_compare_features.py` | So sánh MFCC tĩnh 13 chiều và MFCC động 39 chiều. |
| 13 | `python step13_covariance.py` | So sánh covariance `diag` và `full`. |

**Lưu ý:** Các bước 10-13 là các thí nghiệm riêng lẻ. `step11_states.py`, `step12_compare_features.py` và `step13_covariance.py` huấn luyện các mô hình phụ trong bộ nhớ, không ghi đè mô hình chính trong `models/hmm_models.joblib`.

## 5. Tham số mô hình và xử lý

- Âm thanh: mono, 16 kHz; sau khi đọc sẽ trim silence và pre-emphasis (`coef=0.97`).
- MFCC: 13 hệ số; FFT 512; cửa sổ 25 ms; bước nhảy 10 ms; 26 mel bands; cửa sổ Hamming.
- Đặc trưng: 13 MFCC + 13 Delta + 13 Delta-Delta = 39 giá trị/frame; chuẩn hóa CMVN trên từng utterance.
- HMM chính: 5 states, Gaussian emissions, `covariance_type='diag'`, topology trái-sang-phải.
- Trong bản code cuối đã chạy, `params='mc'` cố định ma trận chuyển trạng thái theo topology đã đặt; các tham số mean và covariance được học. Không mô tả cấu hình này như thể ma trận A đã được cập nhật trong quá trình fit.
- Nhận dạng: chọn nhãn có `model.score(F)` lớn nhất.

## 6. Kết quả đã ghi nhận

- HMM 5 states, MFCC động + covariance diag: **90% (9/10)**.
- Baseline DTW: **100% (10/10)** trong tập test này.
- Thời gian nhận dạng trung bình đo được: HMM **2.342 ms/file**; DTW **1527.531 ms/file**. Đây là thời gian của hai cách cài đặt hiện tại trên máy đã chạy, không phải so sánh tổng quát giữa mọi cách cài đặt HMM và DTW.
- Số trạng thái: 3 states **60%**, 5 states **90%**, 7 states **90%**.
- Đặc trưng: MFCC tĩnh **80%**, MFCC động 39 chiều **90%**.
- Covariance: diag **90%**, full **40%**.
- Robustness: dữ liệu sạch **90%**; thêm nhiễu Gaussian SNR 15 dB **30%**.
- Bigram re-ranking: với corpus minh họa và ngữ cảnh giả định `previous_word='mo'`, nhãn dự đoán không đổi khi thử lambda 0, 0.1, 0.3, 1.0.

Chỉ có 10 file test; mỗi file tương ứng 10 điểm phần trăm accuracy. Không nên suy rộng kết quả ra nhiều người nói hoặc môi trường thực tế.

## 7. Lexicon và Bigram

`lexicon.txt` dùng chuỗi phone tham khảo từ từ điển Vietnamese (Hanoi) MFA. Đây là một quy ước từ điển bên ngoài, không phải bộ phone chính thức do giảng viên cung cấp. Cần nêu rõ nguồn trong báo cáo và dùng cùng một inventory nếu tiếp tục huấn luyện phone/subword HMM.

`corpus_bigram.txt` là corpus minh họa tự tạo. Các kết quả Bigram và MAP re-ranking chỉ minh họa cách đếm xác suất, smoothing và cân bằng điểm; chúng không phải đánh giá language model trên tập ngôn ngữ tự nhiên đại diện.

## 8. File quan trọng để nộp

- `Report_Lab03.pdf` và `Report_Lab03.docx`: báo cáo 4-6 trang (PDF để nộp, DOCX để chỉnh sửa).
- `README.md`: hướng dẫn cài đặt và chạy.
- Các file `.py` cần thiết.
- `lexicon.txt` và `corpus_bigram.txt`.
- `results/`: các bảng CSV và hình kết quả.
- `models/hmm_models.joblib` nếu giảng viên yêu cầu nộp mô hình đã huấn luyện.
- `dataset/` chỉ nộp nếu giảng viên yêu cầu; không đưa bản ghi âm cá nhân lên nơi công khai nếu chưa cần thiết.

## 9. Tài liệu tham khảo

1. *CSE457 - Xử lý âm thanh và tiếng nói. Lab 3: Nhận dạng tiếng nói nâng cao với HMM*, tài liệu thực hành do giảng viên cung cấp.
2. hmmlearn, *API Reference - GaussianHMM*: https://hmmlearn.readthedocs.io/en/stable/api.html
3. Montreal Forced Aligner, *Vietnamese (Hanoi) MFA Dictionary*: https://mfa-models.readthedocs.io/en/latest/dictionary/Vietnamese/Vietnamese%20(Hanoi)%20MFA%20dictionary%20v2_0_0a.html
