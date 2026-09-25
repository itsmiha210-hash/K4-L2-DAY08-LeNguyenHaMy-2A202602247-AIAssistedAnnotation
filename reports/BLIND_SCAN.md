# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 18

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe ở làn giữa di chuyển hướng về phía camera có đèn pha rọi sáng loang trên mặt đường bê tông ướt: vệt phản xạ kéo dài dễ làm AI nhầm lẫn và vẽ box bao trùm cả vệt sáng thay vì ôm sát cản trước xe theo GUIDELINE_LABEL.md.
2. Xe tối màu ở góc dưới bên phải bị mép ảnh cắt mất một phần thân và các xe ở chân cầu xa xôi chỉ có hai chấm đèn mờ: độ tương phản rất thấp chìm vào nền đêm nên AI rất dễ bị False Negative (bỏ sót xe).

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
