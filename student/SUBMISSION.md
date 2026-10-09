# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyen Phuc Huy
- MSSV: 2A202602911
- Email: huy7a2004@gmail.com
- Link repo (fork): https://github.com/Yuhnguyn/K4-L2L3-DAY23-NguyenPhucHuy-2A202602911-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 327ce4c

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`; `[0, 198]`; `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`; `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision `0.9701`, recall `0.7004`; tp `519`, fp `16`, fn `222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse `0.1503` m; matches `502`; sum_sq_err `11.3436` m²; ghost `0`; missed_gt `239`; mean_confirmed `2.5226`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse `0.1359` m; matches `502`; sum_sq_err `9.2668` m²; ghost `0`; missed_gt `239`; mean_confirmed `2.5226`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Hai mode dùng **cùng** detector nên `detection` giống hệt (tp 519, fp 16, fn 222) và do đó `matches = 502`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226` giống nhau. Khác biệt **chỉ** nằm ở RMSE: fused `0.1359` m < lidar `0.1503` m (tổng bình phương sai số giảm `11.3436 → 9.2668` m², tức `−0.0145` m RMSE) vì update camera 2D (AssocC) tinh chỉnh thêm vị trí cho các track nằm trong FOV FRONT. Đọc đủ bộ số: `ghost = 0` nên `precision_track = matches/(matches+ghost) = 502/502 = 1.00`; `coverage = matches/det_tp = 502/519 = 0.967` — 239 `missed_gt_frames` đến từ **detector bỏ sót** (recall 0.70), không phải tracker xoá nhầm (ghost = 0). Vì `rmse_fused − rmse_lidar = −0.0145 ≤ 0.05` m nên camera **không làm tracking xấu đi**. Lưu ý camera ở đây là tâm hộp GT FRONT có nhiễu seeded, không phải camera detector, nên đây không phải bằng chứng về chất lượng một perception system độc lập với GT.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   LiDAR: `z` là vị trí 3D `(x, y, z)` trong hệ cảm biến, `R = diag(σ_lidar_x², σ_lidar_y², σ_lidar_z²)` (3×3, mét²). `h(x) = R_rot·p + t` là **tuyến tính** nên `H` là hằng 3×6 (3 cột vị trí bằng ma trận xoay, 3 cột vận tốc bằng 0). Camera: `z = [u, v]` là pixel, `R = diag(σ_cam_i², σ_cam_j²)` (2×2, pixel²). `h(x)` là **phi tuyến** (pinhole): `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`, nên `H` là Jacobian 2×6 theo quy tắc chuỗi (đạo hàm phép chiếu × ma trận xoay) và **phụ thuộc trạng thái**.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   Mahalanobis `d² = γᵀ S⁻¹ γ` (với `S = H P Hᵀ + R`) tính khoảng cách **theo bất định** của cả dự báo (`P`) lẫn đo (`R`), rồi cổng χ² loại cặp có `d² > χ²_{0.995, dim_meas}`. Khác Euclidean: Euclidean chỉ so vị trí, còn Mahalanobis chia theo phương sai — hai điểm xa nhưng `P` lớn vẫn có thể nằm trong cổng, còn hai điểm gần nhưng `P` nhỏ vẫn có thể bị loại. Nhờ vậy tránh gán nhầm khi track/đo gần nhau về Euclid nhưng không hợp lý về thống kê (bất định lớn).

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**
   Là **track-then-fuse**: chỉ **một** tracker. Mỗi frame, `KF.predict()` chạy **một lần**, rồi `associate_and_update(..., lidar)` (AssocL) update LiDAR, rồi `associate_and_update(..., camera)` (AssocC) update camera — camera chỉ tinh chỉnh EKF, không tạo/xác nhận/xoá track. Trên log, mỗi record có cùng `frame` cho cả `mode=lidar` và `mode=fused`; `det_*`/`valid_gt` (detection) và `confirmed`/`matches` (tracking) là hai tầng riêng, và `compare` chứa record của cả hai mode trên cùng segment/frame.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   `h(x)` bị lệch nên `γ = z − h(x)` có **bias hệ thống**, không còn zero-mean. Sai intrinsics (tiêu cự/tâm) làm `γ` lệch **tỉ lệ** theo khoảng cách và theo vị trí ảnh; sai extrinsic (R, t) làm `γ` lệch theo hướng (xoay/góc nhìn). Hệ quả: Mahalanobis của các cặp camera tăng, cặp bị cổng χ² **loại thường xuyên**; nếu vẫn update, trạng thái vị trí bị kéo lệch → RMSE fused tăng và ghost tăng so với lidar-only.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao lidar quyết định score/init/delete còn camera chỉ EKF update?**
   Vì cuối mỗi lượt phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`, và `manage_tracks` dựa vào `sensor` để quyết định vòng đời. Khi `meas_list` **rỗng** vẫn phải chạy lượt này để trừ score cho các lidar miss (hoặc không làm gì nếu là camera), nên phải truyền `sensor` tường minh thay vì suy ra từ danh sách đo rỗng. Chỉ LiDAR cho đo vị trí 3D đủ tin cậy để **khởi tạo** track và **đánh giá tồn tại** (score/confirm/delete); camera chỉ cho đo 2D, không đủ để tạo/xác nhận/xoá, chỉ tinh chỉnh trạng thái → camera **không** cộng/trừ score, không tạo, không xoá.

6. **Điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**
   LiDAR hit: `score += 1/window` (tối đa 1). Khi `score > confirmed_threshold (0.8)` → **confirmed**. Track đã confirmed **không hạ trạng thái** chỉ vì một miss (miss chỉ trừ score `1/window`). **Xoá** (chỉ lượt lidar, điều kiện OR): `P[0,0]` hoặc `P[1,1] > max_P (9)` → xoá bất kể score; confirmed có `score < delete_threshold (0.6)` → xoá; chưa confirmed có `score ≤ 0` → xoá. Camera không đánh giá/xoá track.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (AI coding agent) qua OpenCode.
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ viết 4 hàm Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) theo công thức trong đề, cấu hình môi trường Python 3.12, và chạy `fusion-run-lab`.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy `pytest student/tests -q` (128 passed, không `failed`/`xfailed`), đối chiếu từng công thức EKF/Mahalanobis/pinhole/vòng đời với `docs/HUONG_DAN_KY_THUAT.md` và `CHECKPOINTS.md`, rồi chạy `fusion-run-lab --fusion compare --seed 0` trên segment Waymo và đọc `metrics.json` + `grade_run.log` (kiểm tra `matches+ghosts==confirmed`, `matches+misses==valid_gt`).

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
