# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603497
- **Họ và tên:** Nguyễn Thị Thanh Bình
- **SEED:** 5586

## 1. Bằng chứng

| Tệp | Có? | Ghi chú |
| --- | --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ | Đã cung cấp notebook có output; ô báo cáo cuối ghi 13/15 kiểm tra đạt. |
| [`lab1_report.json`](lab1_report.json) — do ô “Báo cáo cuối” sinh ra | ☐ | Output notebook thông báo đã lưu, nhưng chưa có tệp JSON đính kèm để xác minh/nộp. |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ | Chưa có ảnh viết tay đính kèm; cần viết và chụp thực tế. |

Dữ liệu sau loại bỏ tin trùng chính xác: **5.169 tin** (4.516 ham, 653 spam). Chia dữ liệu: **train 3.619**, **validation 775**, **test 775**; bộ SUB gồm **40 tin**. Baseline luôn dự đoán ham: 4.516 / 5.169 ≈ **0,87**. Kết quả tính tay đã được notebook kiểm tra: sigmoid(0) = 0,50; F1 ví dụ = 0,73; cosine ví dụ = 0,60; chi phí LLM giả định = 10,50 USD/ngày cho 1 triệu SMS.

## 2. Kết quả chính

Bảng trích từ ô tổng hợp Phần 5 của notebook (SEED = 5586):

| Phương pháp | Precision | Recall | F1 test | F1 SUB 40 | Train (s) | Tin/s (cả lô) | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 1.000 | 0.845 | 0.916 | 0.788 | 9.389 | 18,290.837 | 0.127 | 0.00 |
| 2. BoW + MLP | 1.000 | 0.948 | **0.974** | 0.974 | 7.482 | 37,290.348 | **0.336** | 0.00 |
| 3a. MiniLM + kNN | 0.877 | 0.959 | 0.916 | 0.950 | 78.626 | 19.593 | 53.546 | 0.00 |
| 3b. MiniLM + Logistic Regression | 0.941 | 0.979 | 0.960 | **1.000** | 79.058 | 35.339 | 35.743 | 0.00 |
| 3c. MiniLM + MLP head | 0.921 | 0.959 | 0.939 | 0.950 | 79.532 | 36.017 | 160.772 | 0.00 |

**Nhận xét:** BoW + MLP đạt F1 cao nhất trên toàn bộ tập test (0,974). MiniLM + LogReg đạt F1 = 1,000 trên SUB nhưng tập này chỉ có 40 tin, nên không đủ để kết luận nó luôn tốt hơn. Các con số USD = 0 là **không phát sinh phí API** trong thí nghiệm, không có nghĩa chi phí máy chủ hay điện bằng 0. Phần LLM không có kết quả đánh giá thực tế vì `HAS_LLM = False`; notebook cũng ghi 2 nhóm kiểm tra API/prompt chưa đạt.

## 3. Dự đoán và thực tế

**(1) Tìm kiếm ngữ nghĩa — `emb_search`.** Tôi đã dự đoán kết quả gần truy vấn nhận thưởng du lịch sẽ nghiêng về **ham**. Thực tế, **cả 6 hàng xóm gần nhất đều là spam**, điểm cosine khoảng 0,523–0,563. Tôi đánh giá thấp khả năng encoder nhận ra ý định nhận giải/đòi phản hồi ngay cả khi truy vấn không dùng đúng toàn bộ từ khóa quen thuộc.

**(2) Spam khó — `hard_eval`.** Tôi dự đoán BoW + MLP sẽ đạt accuracy khoảng **65–75%** trên 50 case khó. Thực tế chỉ đạt **54%** và F1 **0,378**, thấp hơn hẳn dự đoán. Điều này cho thấy điểm test thông thường rất cao không đồng nghĩa với khả năng chống lại tin nhắn ngụy trang, spam giọng tự nhiên và tin hợp lệ có từ khóa giống spam.

**(3) MLP bỏ ReLU — `dl_relu`.** Tôi dự đoán F1 có thể giảm nhẹ khi bỏ `nn.ReLU()`, vì hai lớp Linear liên tiếp tương đương một ánh xạ tuyến tính. Thực tế, mô hình **không ReLU vẫn đạt F1 test = 0,974**, bằng F1 của MLP có ReLU trong lần chạy này. Tôi hiểu thêm rằng phi tuyến tạo *khả năng* mô hình hóa quan hệ phức tạp, nhưng chưa chắc đem lại lợi ích đo được trên bài toán và bộ dữ liệu cụ thể.

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB 40 | F1 50 case khó | Mức giảm F1 |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.788 | 0.378 | 0.410 |
| **BoW + MLP** | **0.974** | **0.378** | **0.596** |
| MiniLM + kNN | 0.950 | 0.724 | 0.226 |
| MiniLM + LogReg | 1.000 | 0.667 | 0.333 |
| MiniLM + MLP head | 0.950 | 0.667 | 0.283 |

**BoW + MLP tụt nhiều nhất** theo so sánh F1 SUB với F1 bộ khó: **−0,596**. Đây là hai bộ có phân phối và số lượng khác nhau, vì vậy mức giảm phản ánh stress test, không phải ước lượng suy giảm hiệu năng ngoài đời. Nhóm **spam giọng tự nhiên (`natural_spam`)** là điểm yếu nổi bật: các case #12, #13, #14, #16 và #19 đều bị cả năm mô hình đã chạy đoán sai.

- **Case #14 — spam giọng tự nhiên:** “Hi, I'm working from home and making about £300 a day, happy to show you how...” (nhãn thật: spam). Tin giả dạng câu chuyện cá nhân về việc làm tại nhà, không nhất thiết dùng các từ khóa spam rõ ràng, nên dễ bị dự đoán thành ham. Encoder cũng có thể nhận gần các cuộc trò chuyện đời thường.
- **Case #41 — tin giao dịch hợp lệ:** “Your OTP for login is 482913. Do not share this code...” (nhãn thật: ham). MiniLM + MLP head đoán sai. Từ ngữ như OTP, cảnh báo bảo mật và yêu cầu liên hệ hỗ trợ có thể trông giống nội dung giả mạo ngân hàng, nhưng trong bộ test này đây là thông báo giao dịch hợp lệ.

Những lỗi này cho thấy cả việc **bỏ sót spam (FN)** và **chặn nhầm ham (FP)** đều đáng quan tâm. Bộ 50 tin viết tay thích hợp để tìm điểm yếu nhưng không đại diện đầy đủ cho lưu lượng SMS thực tế. Không có kết quả nhóm `prompt_injection` của LLM vì Phần 4 chưa chạy.

## 5. Thử thách cá nhân

**Đề #1 (SEED 5586):** Thay đổi ngưỡng quyết định của BoW + LogReg từ 0,5 sang 0,2 và 0,8; so sánh precision, recall, F1 và chọn ngưỡng phù hợp khi ngân hàng sợ chặn nhầm OTP.

**Cách làm:** Giữ nguyên mô hình đã train, lấy xác suất `predict_proba(vectorize(texts_te, vocab), w, b)` rồi đổi nhãn bằng phép so sánh `probs >= threshold`; tính `metrics(y_te, y_pred)`, đồng thời đếm FP và FN.

| Ngưỡng | Precision | Recall | F1 | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.948 | **0.938** | **0.943** | 5 | 6 |
| 0.5 | **1.000** | 0.845 | 0.916 | **0** | 15 |
| 0.8 | **1.000** | 0.670 | 0.802 | **0** | 32 |

**Giải thích:** Hạ ngưỡng xuống 0,2 bắt được thêm spam (Recall tăng, FN giảm) nhưng chặn nhầm **5 tin ham**; tăng lên 0,8 không chặn nhầm ham nào trên tập test này, nhưng bỏ sót nhiều spam hơn (**32 FN**). Nếu mục tiêu ưu tiên bảo vệ OTP và phải chọn một trong ba ngưỡng, tôi nghiêng về **0,8** vì quyết định chặn thận trọng hơn; tuy nhiên, cả **0,5 và 0,8 đều có FP = 0** trong tập test hiện tại, nên chưa có bằng chứng 0,8 thực sự tốt hơn trên tin OTP. Cần kiểm tra riêng tập OTP và lưu lượng thực trước khi triển khai.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày (trung bình khoảng **116 SMS/giây**) và yêu cầu **p95 < 50 ms/tin**, các mô hình đạt giới hạn trong notebook gồm **BoW + LogReg (0,127 ms)**, **BoW + MLP (0,336 ms)** và **MiniLM + LogReg (35,743 ms)**. MiniLM + kNN (53,546 ms) và MiniLM + MLP head (160,772 ms) không đạt.

Tôi chọn **BoW + MLP làm lớp phân loại chính** vì F1 test = **0,974**, cao nhất trong năm mô hình, cùng p95 = **0,336 ms/tin** và thông lượng thử nghiệm cả lô khoảng **37.290 tin/giây**. Tôi chấp nhận rằng model vẫn có thể bỏ sót spam giọng tự nhiên và cần giám sát lỗi trên dữ liệu thực; benchmark Colab không thay thế kiểm thử tải/đỉnh lưu lượng trên hạ tầng triển khai.

**Đề xuất router:** Trước tiên dùng BoW + MLP cho tất cả SMS. Các tin có xác suất spam **dưới 0,3** được ưu tiên xem là ham; các tin **trên 0,7** được ưu tiên xem là spam; các tin **0,3–0,7** hoặc thuộc dạng rủi ro (ví dụ nghi giả mạo giao dịch) được chuyển đến tầng kiểm tra bổ sung như MiniLM + LogReg, quy tắc xác minh nguồn gửi, hoặc LLM. Lợi ích là giữ phần lớn lưu lượng trên mô hình nhanh và rẻ, trong khi dành phân tích sâu cho phần khó. Nhược điểm là có những tin sai nhưng mô hình vẫn tự tin; router theo ngưỡng không bắt được tất cả, và nhánh LLM có thể vi phạm SLA nếu đặt trực tiếp trên đường xử lý đồng bộ.

**Chi phí LLM giả định:** Theo giá **0,10 USD/1 triệu input token** và **0,50 USD/1 triệu output token**, mỗi SMS dùng 100 input + 1 output token: toàn bộ **10 triệu SMS/ngày = 105 USD/ngày = 3.150 USD/30 ngày**. Nếu chỉ 5% SMS chuyển qua LLM, chi phí token giả định là **157,50 USD/30 ngày**. Đây là phép tính theo đề, chưa bao gồm hạ tầng, token bổ sung và chưa có đo p95 LLM.

## 7. AI log và điều đã học

Nhật ký dưới đây tóm tắt các nhóm câu hỏi đã trao đổi với AI trong quá trình làm bài; phần “tôi sửa/kiểm tra” chỉ ghi những gì có thể đối chiếu với notebook, không giả định các thao tác không được ghi lại.

| Nội dung hỏi AI | AI hỗ trợ | Kiểm tra/điều chỉnh thể hiện trong notebook |
| --- | --- | --- |
| Phần 1: vectorize, sigmoid, gradient descent, precision/recall/F1 | Giải thích công thức và đưa ví dụ code | Các bài kiểm tra 1.1–1.4 đều đạt; dùng chung dữ liệu train/val/test. |
| Phần 2: xây MLP và 5 bước train | Gợi ý `zero_grad`, forward, BCEWithLogitsLoss, backward, `step` | Các bài kiểm tra 2.1–2.2 đạt; thực nghiệm thêm mô hình bỏ ReLU. |
| Phần 3: cosine, top-k, kNN | Gợi ý chuẩn hóa vector, nhân ma trận và bỏ phiếu | Các bài kiểm tra 3.1–3.3 đạt; thực hiện tìm hàng xóm gần nghĩa và so sánh encoder. |
| Phân tích biểu đồ Pareto, stress test và chọn mô hình | Hỗ trợ giải thích cách đọc các chỉ số và trade-off | So sánh lại bằng số liệu p95, F1 trên tập SUB và 50 case khó do notebook in ra. |
| Lỗi `HAS_LLM` và Phần 4 | Gợi ý đặt `HAS_LLM = False` để không gọi API | Notebook không có số đo LLM; hai nhóm kiểm tra prompt/API còn chưa đạt. |
| Challenge #1: đổi threshold | Gợi ý code tính ba ngưỡng và cách hiểu FP/FN | Có bảng thực nghiệm 0,2/0,5/0,8 với precision, recall, F1, FP, FN. |

**Điều tôi học được:** Trước đây tôi nghĩ mô hình phức tạp hơn thì chắc chắn tốt hơn, nhưng trên test thường BoW + MLP lại đạt F1 cao hơn các mô hình embedding. Tôi cũng hiểu rằng accuracy/F1 cao trên dữ liệu thông thường vẫn có thể che giấu nhiều lỗi ở các trường hợp khó; đặc biệt, lựa chọn threshold cần dựa vào hậu quả của FP và FN trong tình huống sử dụng thực tế. Cuối cùng, độ trễ, chi phí và khả năng giám sát phải được đánh giá cùng với chất lượng dự đoán.

---

**Tình trạng hoàn thành:** Notebook có output, **13/15** kiểm tra đạt; Phần 4 (LLM) chưa hoàn thành. Cần bổ sung tệp `lab1_report.json` và ảnh viết tay `hand_calc.jpg` trước khi nộp đủ bộ; không đánh dấu là đã có khi chưa đính kèm.
