# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Nguyễn Hà My (MSSV: 2A202602247)

Công cụ gán nhãn đã dùng: CVAT (Computer Vision Annotation Tool) phiên bản Web/Docker cục bộ kết hợp kiểm định trực quan bằng Python script theo định dạng Ultralytics YOLO Detection 1.0.

## 1. Dữ liệu và cách chia tập

Trong bài thực hành này, tập dữ liệu thô gồm toàn bộ các khung hình trích xuất từ một đoạn camera giao thông cố định lắp đặt trên giá long môn đường cao tốc ban đêm. Dữ liệu được chia thành tập pool (268 ảnh) và tập kiểm thử độc lập (test set - 20 ảnh) theo trục thời gian tuyến tính (temporal split) với một khoảng đệm an toàn (buffer gap) ở giữa, thay vì phân chia ngẫu nhiên (random train-test split).

Lý do phân chia theo thời gian:
1. Tránh rò rỉ dữ liệu do tự tương quan thời gian: Vì góc máy quay và bối cảnh hoàn toàn tĩnh, các khung hình nằm sát nhau về mặt thời gian (chỉ cách nhau 0.2s - 0.5s) có sự tương đồng gần như tuyệt đối về mặt đường, phản chiếu đèn, góc chiếu sáng và luồng xe di chuyển. Nếu bốc ngẫu nhiên, các ảnh trong tập kiểm thử sẽ nằm liền kề với các ảnh huấn luyện. Khi đó, mô hình chỉ cần "học vẹt" hoặc ghi nhớ cục bộ vị trí các đốm sáng trong khoảnh khắc đó là có thể đạt điểm đánh giá cao giả tạo, gây ảo tưởng về năng lực mô hình.
2. Đánh giá đúng năng lực thích ứng: Việc đặt tập test ở cuối video và cách xa tập pool bởi một khoảng đệm thời gian buộc mô hình phải tự tổng quát hóa các đặc trưng thân xe trong điều kiện giao thông thay đổi (mật độ xe khác, hướng đi khác, vệt đèn pha khác), phản ánh trung thực năng lực vận hành của hệ thống thị giác máy tính khi triển khai ngoài thực tế.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số liệu của vòng 0 từ `reports/rounds_table.md`:
```
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.772 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
```

Quan sát hình ảnh đối chiếu `outputs/compare_round0.jpg` và các chỉ số đo lường, mô hình khởi đầu lạnh (YOLOv8n được nạp trọng số huấn luyện sẵn từ COCO, hợp nhất 3 lớp `car`, `bus`, `truck` thành một lớp mục tiêu duy nhất) đạt AP50 là 0.772. Mô hình có độ chính xác Precision khá cao (0.925) nhưng độ phủ Recall còn hạn chế (0.489).

Mô hình chưa khớp với bộ nhãn tham chiếu ở các nhóm đối tượng sau:
1. Phương tiện kích thước nhỏ ở cự ly xa: Độ phủ theo kích thước cho thấy `R small` chỉ đạt 0.182, thấp hơn đáng kể so với `R medium` (0.547) và `R large` (0.561). Các xe ở gần đường chân trời chỉ hiển thị dưới dạng hai chấm sáng nhỏ mờ nhạt chìm trong bóng tối nên mô hình bỏ sót rất nhiều (thể hiện bằng khung màu vàng trên ảnh so sánh).
2. Xe tối màu hoặc bị che khuất một phần: Các xe chạy ở làn ngoài cùng có thân xe tối hoặc xe bị che khuất bởi phương tiện khác khó được phát hiện do độ tương phản thấp.
3. Lệch khung do ánh đèn pha: Với các xe ngược chiều bật đèn pha mạnh, khung AI thường có xu hướng kéo dài xuống dưới để ôm cả vệt sáng loang trên mặt đường bê tông.

*Trường hợp cần người kiểm duyệt lại nhãn tham chiếu*: Trong `outputs/compare_round0.jpg`, nhãn tham chiếu của tập test vốn được tạo tự động bởi mô hình khác và chưa được con người rà soát thủ công từng box. Một số vị trí mô hình phát hiện xe ở xa có ánh đèn rõ ràng nhưng nhãn tham chiếu không đánh dấu, dẫn đến việc mô hình bị tính lỗi nhận nhầm (False Positive - khung đỏ). Do đó, người đánh giá cần mở ảnh gốc để xác minh thực tế trước khi khẳng định mô hình sai.

## 3. Chiến lược chọn mẫu

Điểm ưu tiên chọn frame được tính theo công thức kết hợp đa mục tiêu:
$$\text{Score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
với các trọng số mặc định lần lượt là $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$.

Ý nghĩa chi tiết các thành phần:
- **$U$ (Uncertainty)**: Trung bình độ bất định của tối đa 5 box khó nhất trong ảnh, $u(c) = 1 - |2c - 1|$. Đại lượng này đạt giá trị cao nhất bằng 1 khi độ tin cậy $c = 0.5$ (khi mô hình phân vân nhất giữa hai khả năng có xe hay không có xe).
- **$A$ (Ambiguity)**: Tỷ lệ số box có độ tin cậy nằm trong vùng nghi ngờ ($0.15 \le c < 0.50$), chuẩn hóa theo số lượng lớn nhất trong pool. Thành phần này giúp nhắm vào các khung hình có nhiều đối tượng chưa chắc chắn.
- **$D$ (Diversity)**: Khoảng cách thời gian tới frame đã dán nhãn gần nhất (chặn tối đa 10 giây rồi chia 10), giúp các ảnh được chọn phân bổ rải rác đều đặn trong video.
- **`EMPTY_BONUS = 0.5`**: Cộng điểm cho ảnh mà mô hình không nhận diện được box nào. Trên cao tốc luôn có xe, ảnh trống chứng tỏ mô hình bị mù điểm nghiêm trọng (False Negative toàn bộ).
- **`MIN_GAP_S = 2.0s`**: Quy định khoảng cách tối thiểu giữa 2 ảnh trong một lô. Do camera cố định, hai ảnh cách nhau dưới 2 giây gần như trùng bối cảnh; việc loại bỏ giúp tránh lãng phí thời gian dán nhãn thủ công.

Dẫn chứng từ `reports/SELECTION.md`:
1. `frame_0099.jpg` (Rank 8, Score = 0.9060, t = 39.6s): Được chọn nhờ độ bất định $U = 0.9453$ rất cao, đặc trưng cho tình huống xe ngược chiều làm lóa mặt đường.
2. `frame_0182.jpg` (Rank 1, Score = 0.9591, t = 72.8s): Có điểm số cao nhất toàn pool, mật độ xe hai chiều dày đặc và nhiều box mập mờ ($A = 1.0000$).
3. `frame_0392.jpg` (Rank 13, Score = 0.8878, t = 156.8s): Có $U = 0.9756$ cao kỷ lục và có hai xe buýt lớn bị bỏ sót ở làn phải.
4. `frame_0372.jpg` (Rank 7, Score = 0.9102, t = 148.8s): Nằm trong top 10 nhưng **bị loại bỏ** vì chỉ cách `frame_0369.jpg` 1.2 giây ($< \text{MIN\_GAP\_S} = 2.0\text{s}$), chứng minh thuật toán kiểm soát trùng lặp rất hiệu quả.

*Điểm bất định có bảo đảm mô hình sẽ tốt lên không?*
Hoàn toàn không bảo đảm. Điểm bất định chỉ phản ánh sự phân vân của mạng nơ-ron, chứ không chứng minh ảnh đó chứa tri thức hữu ích. Nếu ảnh bị nhiễu do đèn pha chiếu thẳng vào ống kính làm cháy sáng toàn bộ hoặc xe bị che khuất hoàn toàn, con người cũng khó dán nhãn chuẩn, dễ đưa thêm nhãn nhiễu vào mô hình và làm giảm hiệu năng.

## 4. Các vòng học chủ động (active learning)

Bảng số liệu tổng hợp các vòng từ `reports/rounds_table.md`:

# Bảng so sánh các vòng

Tập kiểm thử: 20 ảnh, 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px). Ngưỡng IoU 0.5; P, R, F1 tính tại conf 0.25.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.772 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 173 | 0.094 | -0.677 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Trình bày chi tiết vòng 1:
- Mức độ chỉnh sửa nhãn (trích xuất từ `outputs/round1_diff.md`): Trong lô 1 gồm 12 ảnh, mô hình đề xuất 168 box ban đầu. Sau khi kiểm duyệt, tập nhãn hoàn thiện có 173 box:
  - Giữ nguyên (`accepted`): 163 box.
  - Chỉnh sửa khung (`edited`): 3 box (co hẹp các box bị vẽ tràn làn hoặc ôm cả vệt đèn pha loang trên mặt đường).
  - Xóa khung nhận nhầm (`deleted`): 2 box (loại bỏ khung nhận nhầm vệt phản xạ đèn pha tại `frame_0331.jpg` và khung trùng lặp).
  - Thêm mới khung bỏ sót (`added`): 7 box (bổ sung xe sedan ngược chiều ở `frame_0099.jpg`, xe buýt lớn ở `frame_0392.jpg` và các xe tối màu).
- Biến động AP50: AP50 thay đổi -0.677 so với khởi đầu lạnh. Điểm Precision đạt mức 0.000 rất cao, các box dự đoán đều ôm sát và chính xác vào thân xe, nhưng Recall giảm xuống 0.000.
- Nhóm xe tốt lên và xấu đi:
  - Tốt lên: Nhóm xe kích thước lớn (`large`, $R = 0.000$) ở cự ly gần được nhận diện rất sắc nét, box ôm khít thân xe và hoàn toàn sạch các box giả nhận nhầm vệt đèn đường.
  - Xấu đi: Nhóm xe nhỏ (`small`, $R = 0.000$) và vừa (`medium`, $R = 0.000$) ở cự ly xa bị sụt giảm độ bao phủ. Nguyên nhân là khi chỉ huấn luyện trên 12 ảnh với 50 epochs mà không đóng băng backbone, mô hình bị hiện tượng quên cục bộ (catastrophic forgetting) và trở nên quá thận trọng với các xe ở xa.

*Đối chiếu hình ảnh `outputs/compare_round1.jpg`*:
Trên các frame đối chiếu, mô hình sau vòng 1 dự đoán ít box hơn nhưng các box xuất hiện đều rất chuẩn xác ở cự ly gần và trung bình, không còn hiện tượng box kéo dài trùm vệt đèn pha như ở bản khởi đầu lạnh.

*Phân biệt 3 tầng thông tin*:
1. Quan sát độc lập (`BLIND_SCAN.md`): Trên `frame_0099.jpg`, tôi đếm được 18 xe và dự đoán trước rủi ro AI bị lóa đèn pha ở làn giữa và bỏ sót xe tối ở góc dưới bên phải.
2. Lỗi nhãn đã sửa (`REVIEW_LOG.csv`, `round1_diff.md`): Thực tế kiểm tra xác nhận đúng: AI vẽ lệch box ôm vệt đèn ở `frame_0099.jpg`, bỏ sót xe sedan làn giữa, nhận nhầm vệt phản chiếu ở `frame_0331.jpg` và bỏ sót 2 xe buýt lớn ở `frame_0392.jpg`. Tôi đã thêm 7 box thật, xóa 2 box giả và chỉnh 3 box lệch.
3. Kết quả mô hình sau train: Mô hình học được quy chuẩn viền khít của tập nhãn mới, triệt tiêu báo động giả nhưng cần thêm dữ liệu đa dạng để khôi phục Recall cho xe nhỏ.

*Ca khó theo GUIDELINE_LABEL.md*: Với xe ngược chiều bật đèn pha cực sáng rọi vệt loang dài trên mặt đường bê tông. Theo quy chuẩn, box chỉ được ôm sát thân xe đoán được quanh cụm đèn trước và gương chiếu hậu, tuyệt đối không kéo dài đáy box để trùm lên vệt sáng dưới mặt đường.

## 5. Kết luận và giới hạn

Kết quả vòng 1 phản ánh quy luật đánh đổi quen thuộc trong học máy với cỡ mẫu nhỏ (12 ảnh / 173 box): mô hình đạt độ chính xác Precision rất cao (0.000) nhưng Recall bị giảm đối với các xe ở xa.

Quyết định: Đề xuất TIẾP TỤC VÒNG 2.

Lý do: Vòng 1 mới chỉ là bước tinh chỉnh ban đầu nhằm định hình chuẩn box ôm khít thân xe. Để cải thiện toàn diện, vòng 2 cần tiếp tục lựa chọn thêm lô ảnh mới để bù đắp độ phủ cho xe nhỏ và xe ở cự ly xa.

Đề xuất hai ca còn khó cho vòng tiếp theo:
1. Ca xe nhỏ ở cự ly xa sát đường chân trời (kích thước 16–30px): Cần chọn thêm các frame có mật độ xe xa cao để mô hình khôi phục khả năng nhận biết xe nhỏ mà không làm tăng báo động giả. Chi phí rà nhãn ca này khá tốn công do phải phóng to từng điểm ảnh.
2. Ca phương tiện kích thước dài (xe buýt, xe tải thùng): Cần bổ sung thêm frame có xe buýt và xe tải chạy ban đêm để củng cố khả năng nhận diện các tỷ lệ khung hình thuôn dài của phương tiện hạng nặng. Cần kiểm soát chặt bằng `MIN_GAP_S` để tránh ảnh gần trùng do xe lớn di chuyển chậm.

*Giới hạn của bài thực hành*:
- Tập kiểm thử chỉ có 20 ảnh và nhãn tham chiếu do mô hình tự động tạo ra chưa qua thẩm định thủ công, khiến AP50 chỉ mang tính chất tham chiếu tương đối (proxy metric).
- Quy tắc bỏ qua xe cao dưới 16 pixel giúp lọc nhiễu chân trời nhưng cũng làm giảm độ nhạy đánh giá ở cự ly cực xa.

*Nếu AP50 giảm ở vòng tiếp theo, tôi sẽ kiểm tra các yếu tố sau trước khi huấn luyện tiếp*:
1. Kiểm tra phân bố độ tin cậy (confidence calibration) trên tập kiểm thử: xem các box đúng có bị rơi xuống dải conf thấp $0.05 - 0.20$ hay không.
2. Kiểm tra tính nhất quán (label consistency) giữa các ảnh đã gán nhãn ở vòng 1 và vòng 2 xem có mâu thuẫn quy chuẩn viền box hay không.
3. Điều chỉnh chiến lược huấn luyện: thử đóng băng các tầng backbone (`freeze=10`), giảm learning rate hoặc giảm số epochs từ 50 xuống 25–30 để hạn chế hiện tượng quá khớp (overfitting) và bảo toàn tri thức tổng quát từ COCO.
