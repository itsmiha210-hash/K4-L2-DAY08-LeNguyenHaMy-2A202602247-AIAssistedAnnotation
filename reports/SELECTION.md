# Vì sao chọn lô này?

Trong 50 dòng đứng đầu của `outputs/selection_round1.csv`, nếu chỉ có ngân sách thời gian để rà soát 5 ảnh, tôi ưu tiên lựa chọn 5 frame sau:
1. **`frame_0099.jpg`** (Rank 8, Score = 0.9060, t = 39.6s, U = 0.9453, A = 0.7778): Nằm ở khoảng đầu video (t = 39.6s) với độ bất định U rất cao (0.9453). Ảnh có tình huống xe ngược chiều bật đèn pha mạnh làm lóa mặt đường bê tông ướt khiến AI bỏ sót xe sedan lớn ở giữa và xe sát dải phân cách.
2. **`frame_0182.jpg`** (Rank 1, Score = 0.9591, t = 72.8s, U = 0.9182, A = 1.0000, 18 box mơ hồ): Frame có điểm tổng hợp cao nhất tập pool; bối cảnh giao thông phức tạp hai chiều với nhiều xe tối màu và khung AI bị vẽ tràn làn.
3. **`frame_0331.jpg`** (Rank 5, Score = 0.9153, t = 132.4s, 47 boxes, A = 1.0000): Mật độ giao thông cao, có hiện tượng phản chiếu ánh đèn tạo ra các box nhận nhầm giả (False Positive) cần người kiểm duyệt xử lý.
4. **`frame_0380.jpg`** (Rank 3, Score = 0.9168, t = 152.0s, U = 0.9336, A = 0.8333, 40 box): Thuộc giai đoạn cao điểm cuối video với lưu lượng xe lớn, giúp mô hình học cách phân biệt các xe chạy san sát nhau.
5. **`frame_0392.jpg`** (Rank 13, Score = 0.8878, t = 156.8s, U = 0.9756): Độ bất định U cao kỷ lục; đặc biệt xuất hiện hai xe buýt lớn màu trắng đỏ ở làn phải mà AI bỏ sót hoàn toàn. Rà frame này cung cấp mẫu học rất tốt cho lớp `car` ở nhóm xe thương mại dài.

*Cân nhắc ảnh gần trùng và frame trống*:
- `frame_0372.jpg` (Rank 7, Score = 0.9102, t = 148.8s) có điểm rất cao nhưng bị thuật toán loại bỏ vì chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây, vi phạm ngưỡng `MIN_GAP_S = 2.0s`. Do góc quay camera cố định, hai khung hình này gần như giống hệt nhau, việc gán nhãn cả hai sẽ gây lãng phí công sức mà không đem lại tri thức mới.
- Đối với frame không có box nào (`empty = True`), công thức cộng thêm `EMPTY_BONUS = 0.5`. Do cao tốc luôn có xe lưu thông, frame trống đồng nghĩa với việc mô hình bị mù điểm hoàn toàn, cần được người rà soát can thiệp gấp.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
1. **`frame_0099.jpg`**: Rank 8 trong CSV, thể hiện luồng giao thông với ánh đèn pha ngược chiều rọi sáng mặt đường, có xe sedan lớn chìm trong ánh sáng.
2. **`frame_0182.jpg`**: Xếp Rank 1 trong CSV với 18 box mơ hồ ($A = 1.0000$). Trên ảnh contact sheet, mật độ xe đan xen phức tạp giữa xe ngược chiều đèn trắng và xe cùng chiều đèn đỏ.
3. **`frame_0392.jpg`**: Xếp Rank 13 trong CSV với $U = 0.9756$. Trên ảnh ghép contact sheet, quan sát rõ 2 thân xe buýt lớn ở làn phải phía xa bị mô hình bỏ sót.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- **`frame_0372.jpg`** (Rank 7, Score = 0.9102, t = 148.8s): Nằm trong top điểm cao nhất nhưng **không được chọn** do quá gần `frame_0369.jpg` (t = 147.6s, chênh lệch 1.2s < MIN_GAP_S 2.0s). Quy tắc lọc trùng lặp này rất hợp lý để tối ưu hóa chi phí dán nhãn của con người.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Phép chọn chỉ đo lường **độ phân vân nội tại** của mô hình (confidence quanh 0.5), hoàn toàn **không phản ánh được mô hình đúng hay sai trong thực tế**.
- Mô hình có thể dự đoán sai nhưng với độ tự tin cực cao (overconfident False Positive, ví dụ biển báo phản quang nhận nhầm thành xe với conf > 0.95), khi đó $u pprox 0$ và thuật toán sẽ bỏ qua frame đó.
- Ngược lại, điểm cao có thể rơi vào ảnh bị nhòe mờ, nhiễu hạt nặng mà mắt người cũng không thể xác định ranh giới thân xe, khiến việc dán nhãn đưa thêm nhiễu vào tập huấn luyện.
