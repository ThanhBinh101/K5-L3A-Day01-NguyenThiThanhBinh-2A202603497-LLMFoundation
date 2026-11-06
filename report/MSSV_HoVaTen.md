# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603497
- **Họ và tên:** Nguyễn Thị Thanh Bình
- **SEED** (in ở ô đầu notebook): 5586

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

| Phương pháp | Precision | Recall | F1 test | F1 SUB 40 | Train (s) | Tin/s (cả lô) | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 1.000 | 0.845 | 0.916 | 0.788 | 9.389 | 18,290.837 | 0.127 | 0.00 |
| 2. BoW + MLP | 1.000 | 0.948 | **0.974** | 0.974 | 7.482 | 37,290.348 | **0.336** | 0.00 |
| 3a. MiniLM + kNN | 0.877 | 0.959 | 0.916 | 0.950 | 78.626 | 19.593 | 53.546 | 0.00 |
| 3b. MiniLM + Logistic Regression | 0.941 | 0.979 | 0.960 | **1.000** | 79.058 | 35.339 | 35.743 | 0.00 |
| 3c. MiniLM + MLP head | 0.921 | 0.959 | 0.939 | 0.950 | 79.532 | 36.017 | 160.772 | 0.00 |


## 3. Dự đoán và thực tế

1) Tìm kiếm ngữ nghĩa — emb_search. Tôi đã dự đoán kết quả gần truy vấn nhận thưởng du lịch sẽ nghiêng về ham. Thực tế, cả 6 hàng xóm gần nhất đều là spam, điểm cosine khoảng 0,523–0,563. Tôi đánh giá thấp khả năng encoder nhận ra ý định nhận giải/đòi phản hồi ngay cả khi truy vấn không dùng đúng toàn bộ từ khóa quen thuộc.

(2) Spam khó — hard_eval. Tôi dự đoán BoW + MLP sẽ đạt accuracy khoảng 65–75% trên 50 case khó. Thực tế chỉ đạt 54% và F1 0,378, thấp hơn hẳn dự đoán. Điều này cho thấy điểm test thông thường rất cao không đồng nghĩa với khả năng chống lại tin nhắn ngụy trang, spam giọng tự nhiên và tin hợp lệ có từ khóa giống spam.

(3) MLP bỏ ReLU — dl_relu. Tôi dự đoán F1 có thể giảm nhẹ khi bỏ nn.ReLU(), vì hai lớp Linear liên tiếp tương đương một ánh xạ tuyến tính. Thực tế, mô hình không ReLU vẫn đạt F1 test = 0,974, bằng F1 của MLP có ReLU trong lần chạy này. Tôi hiểu thêm rằng phi tuyến tạo khả năng mô hình hóa quan hệ phức tạp, nhưng chưa chắc đem lại lợi ích đo được trên bài toán và bộ dữ liệu cụ thể.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

đối với part 1-3, tôi dùng AI để check syntax và giải thích các định nghĩa chưa hiểu
task 5 và bonus, tôi vibe code, nhờ AI giải thích chi tiết
