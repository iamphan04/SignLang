"""
KSL Data Collector — Thu thập dữ liệu ký hiệu tay tiếng Hàn (지문자)
=======================================================================
Pipeline giống hệt inference.py: BGR → RGB → padding → MediaPipe → normalize → CSV

Phím bấm:
  SPACE         : Lưu 1 mẫu (khi đang xem live) HOẶC bỏ qua ảnh review
  S             : Lưu mẫu (thay thế SPACE nếu cần)
  ← / →         : Chuyển ký hiệu trước / sau
  Số 0-9 / chữ  : Chọn nhanh index ký hiệu trong danh sách
  D             : Xoá mẫu cuối cùng đã lưu
  R             : Reset đếm về 0 (không xoá CSV)
  C             : Xem ảnh đã lưu (chế độ review)
  Q / Esc       : Thoát và lưu CSV
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# Đường dẫn
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import add_image_padding, normalize_landmarks  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "ksl_collected"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# ─────────────────────────────────────────────
# 12 ký hiệu KSL Jimunja được chọn
# Nguồn: 국립국어원 한국수어사전 (sldict.korean.go.kr)
# ─────────────────────────────────────────────
KSL_LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ", "ㅏ", "ㅣ"]
KSL_NAMES  = ["기역", "니은", "디귿", "리을", "미음", "비읍",
              "시옷", "이응", "지읒", "히읗", "아(ㅏ)", "이(ㅣ)"]
KSL_TIPS   = [
    "Ngon tro dung thang, gap nhe tao goc goc ㄱ",
    "Ngon cai (ngang) + ngon tro (doc) → hinh L",
    "Ngon cai + tro + giua tao hinh chu nhat ㄷ",
    "Ngon tro + giua uon cong nhe → duong cong ㄹ",
    "3 ngon cham dau nhau tao hinh vuong ㅁ",
    "4 ngon thang dung chum lai (tru ngon cai)",
    "Ngon tro + giua → hinh V / ㅅ (hướng xuong)",
    "Ngon cai + tro tao vong tron (OK sign)",
    "3 ngon cham dau tao hinh ㅈ (nhu ㅅ + giua)",
    "Ban tay mo rong tao hinh tron lon",
    "Chi ngon cai duoi sang phai (nhu chu L ngang)",
    "Chi ngon tro duoi thang dung",
]

# ─────────────────────────────────────────────
# Feature columns (42 = 21 landmarks × x,y)
# ─────────────────────────────────────────────
FEATURE_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]
CSV_HEADER   = ["person_id", "label", "label_name", "session", "timestamp"] + FEATURE_COLS

# ─────────────────────────────────────────────
# Màu sắc HUD
# ─────────────────────────────────────────────
CLR_GREEN  = (80,  230, 110)
CLR_CYAN   = (60,  210, 250)
CLR_YELLOW = (50,  220, 220)
CLR_RED    = (60,  60,  230)
CLR_GRAY   = (180, 180, 180)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (30,  30,  30)


# ════════════════════════════════════════════════════════
# Hàm tiện ích
# ════════════════════════════════════════════════════════

def build_detector() -> vision.HandLandmarker:
    """Khởi tạo MediaPipe Hand Landmarker — giống hệt inference.py."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Thiếu model: {MODEL_PATH}")
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def process_frame(detector, frame_bgr):
    """
    Chạy toàn bộ pipeline trên 1 frame:
      BGR → RGB → padding 25% → MediaPipe → normalize → 42 features

    Trả về (status, features, display_points):
      status ∈ {"ok", "no_hand", "multiple_hands", "invalid"}
      features: np.ndarray (42,) float32 hoặc None
      display_points: np.ndarray (21,2) pixel coords trong frame gốc hoặc None
    """
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, pad_x, pad_y = add_image_padding(image_rgb)
    ph, pw = padded.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
    result  = detector.detect(mp_img)
    count   = len(result.hand_landmarks)

    if count == 0:
        return "no_hand", None, None
    if count != 1:
        return "multiple_hands", None, None

    pts_norm = np.asarray([[p.x, p.y] for p in result.hand_landmarks[0]], dtype=np.float64)
    try:
        features = normalize_landmarks(pts_norm, pw, ph)
        if features.shape != (42,) or not np.isfinite(features).all():
            raise ValueError
    except ValueError:
        return "invalid", None, None

    # Chuyển về tọa độ pixel của frame GỐC (không padding)
    display_pts = pts_norm * np.array([pw, ph]) - np.array([pad_x, pad_y])
    return "ok", features, display_pts


def draw_hand(frame, display_pts):
    """Vẽ 21 landmark và xương bàn tay lên frame."""
    chains = ((0,1,2,3,4),(0,5,6,7,8),(5,9,10,11,12),
              (9,13,14,15,16),(13,17,18,19,20),(17,0))
    pts = np.rint(display_pts).astype(int)
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (200, 220, 20), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, tuple(p), 4, CLR_CYAN, -1, cv2.LINE_AA)
        if i == 0:
            cv2.putText(frame, "0", tuple(p + [6, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)


def build_canvas(frame, label_idx, counts, status, person_id, session, target):
    """Ghép frame + HUD panel dưới."""
    h, w = frame.shape[:2]
    panel_h = 150
    canvas_w = max(740, w)
    canvas = np.zeros((h + panel_h, canvas_w, 3), dtype=np.uint8)
    canvas[: h, : w] = frame

    # ── Thanh tiến độ cho ký hiệu hiện tại ──────────────────
    cur_label = KSL_LABELS[label_idx]
    cur_name  = KSL_NAMES[label_idx]
    count_cur = counts.get(cur_label, 0)
    bar_w = canvas_w - 24
    filled = int(bar_w * min(count_cur, target) / target)
    cv2.rectangle(canvas, (12, h + 8), (12 + bar_w, h + 22), (50, 50, 50), -1)
    bar_color = CLR_GREEN if count_cur >= target else CLR_YELLOW
    if filled > 0:
        cv2.rectangle(canvas, (12, h + 8), (12 + filled, h + 22), bar_color, -1)
    pct = int(100 * min(count_cur, target) / target)
    cv2.putText(canvas, f"{count_cur}/{target} ({pct}%)", (16, h + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_DARK, 1, cv2.LINE_AA)

    # ── Ký hiệu hiện tại ────────────────────────────────────
    # Label lớn (unicode Hangul) — vẽ bằng PIL để hiển thị đúng
    label_txt = f"[{label_idx+1}/{len(KSL_LABELS)}]  {cur_label}  ({cur_name})"
    cv2.putText(canvas, label_txt, (12, h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_CYAN, 2, cv2.LINE_AA)

    # ── Tip cầm tay ──────────────────────────────────────────
    tip = KSL_TIPS[label_idx]
    cv2.putText(canvas, tip, (12, h + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_GRAY, 1, cv2.LINE_AA)

    # ── Status phát hiện bàn tay ─────────────────────────────
    status_txt = {
        "ok":             "Hand OK  — SPACE / S de luu",
        "no_hand":        "Khong thay ban tay — dua tay vao khung",
        "multiple_hands": "Nhieu ban tay — chi de 1 tay",
        "invalid":        "Landmark khong hop le — thu lai",
    }[status]
    status_clr = CLR_GREEN if status == "ok" else CLR_RED
    cv2.putText(canvas, status_txt, (12, h + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_clr, 1, cv2.LINE_AA)

    # ── Tất cả ký hiệu + số mẫu ─────────────────────────────
    summary = "  ".join(
        f"{lbl}:{counts.get(lbl,0)}" for lbl in KSL_LABELS
    )
    cv2.putText(canvas, summary[:90], (12, h + 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_GRAY, 1, cv2.LINE_AA)

    # ── Footer ───────────────────────────────────────────────
    footer = f"Person: {person_id}  |  Session: {session}  |  SPACE=luu  D=xoa  ←→=chuyen  Q=thoat"
    cv2.putText(canvas, footer, (12, h + 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1, cv2.LINE_AA)

    return canvas


def load_existing_counts(csv_path: Path) -> dict:
    """Đếm số mẫu đã có trong CSV theo label."""
    counts = {lbl: 0 for lbl in KSL_LABELS}
    if not csv_path.is_file():
        return counts
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label", "")
            if lbl in counts:
                counts[lbl] += 1
    return counts


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person",   default="p1",
                        help="ID người thu thập (p1, p2, ...)")
    parser.add_argument("--camera",   type=int, default=0,
                        help="Index webcam (thường là 0)")
    parser.add_argument("--backend",  choices=("auto","dshow","msmf"), default="auto",
                        help="Backend OpenCV (thử dshow nếu auto không mở được)")
    parser.add_argument("--target",   type=int, default=250,
                        help="Số mẫu mục tiêu mỗi ký hiệu (mặc định 250)")
    parser.add_argument("--output",   type=Path, default=None,
                        help="Đường dẫn file CSV output (mặc định data/ksl_collected/ksl_<person>.csv)")
    parser.add_argument("--start-label", type=int, default=0,
                        help="Index ký hiệu bắt đầu (0-11)")
    args = parser.parse_args()

    # ── Chuẩn bị output ─────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = args.output or (DATA_DIR / f"ksl_{args.person}.csv")
    session  = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_header = not csv_path.is_file()

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    if write_header:
        writer.writerow(CSV_HEADER)
    csv_file.flush()

    counts = load_existing_counts(csv_path)
    total_saved = sum(counts.values())

    print(f"\n{'='*60}")
    print(f"  KSL Data Collector — Person: {args.person}")
    print(f"  Output: {csv_path}")
    print(f"  Muc tieu: {args.target} mau/ky hieu × 12 = {args.target * 12} mau")
    print(f"  Da co: {total_saved} mau")
    print(f"{'='*60}")
    print("  SPACE / S : Luu mau")
    print("  ← / →     : Chuyen ky hieu")
    print("  D         : Xoa mau cuoi")
    print("  Q / Esc   : Thoat")
    print(f"{'='*60}\n")

    # ── Khởi tạo detector và webcam ─────────────────────────
    detector = build_detector()
    be_map   = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    cap      = cv2.VideoCapture(args.camera, be_map[args.backend])
    if not cap.isOpened():
        print("❌ Khong mo duoc webcam. Thu --camera 1 hoac --backend dshow")
        detector.close()
        csv_file.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    label_idx     = max(0, min(args.start_label, len(KSL_LABELS) - 1))
    last_features = None   # features frame cuối cùng "ok"
    last_row_key  = None   # để undo (D)
    window_name   = "KSL Collector — SPACE=luu  Q=thoat"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 740, 640)

    # Flash state khi lưu thành công
    flash_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Webcam mat frame. Thu lai.")
                time.sleep(0.05)
                continue

            # ── Xử lý pipeline ──────────────────────────────
            status, features, display_pts = process_frame(detector, frame)
            if status == "ok":
                last_features = features

            # ── Vẽ landmark ──────────────────────────────────
            canvas_frame = frame.copy()
            if display_pts is not None:
                draw_hand(canvas_frame, display_pts)

            # ── Flash xanh khi vừa lưu ──────────────────────
            now = time.perf_counter()
            if now < flash_until:
                overlay = np.full_like(canvas_frame, (0, 60, 0), dtype=np.uint8)
                canvas_frame = cv2.addWeighted(canvas_frame, 0.6, overlay, 0.4, 0)

            # ── Dựng HUD ─────────────────────────────────────
            canvas = build_canvas(canvas_frame, label_idx, counts,
                                  status, args.person, session, args.target)
            cv2.imshow(window_name, canvas)

            # ── Phím bấm ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # Thoát
            if key in (ord("q"), ord("Q"), 27):
                break

            # Lưu mẫu (SPACE hoặc S)
            if key in (ord(" "), ord("s"), ord("S")):
                if status == "ok" and last_features is not None:
                    cur_label = KSL_LABELS[label_idx]
                    cur_name  = KSL_NAMES[label_idx]
                    ts        = datetime.now().isoformat(timespec="milliseconds")
                    row       = ([args.person, cur_label, cur_name, session, ts]
                                 + last_features.tolist())
                    writer.writerow(row)
                    csv_file.flush()
                    counts[cur_label] = counts.get(cur_label, 0) + 1
                    last_row_key = (cur_label, csv_path)
                    flash_until  = time.perf_counter() + 0.25
                    n = counts[cur_label]
                    print(f"  ✅ Luu: {cur_label}({cur_name})  →  tong: {n}/{args.target}")
                    # Tự động sang ký hiệu tiếp khi đủ target
                    if n == args.target:
                        print(f"  🎉 Du {args.target} mau cho {cur_label}! Chuyen sang ky hieu tiep...")
                        time.sleep(0.8)
                        label_idx = (label_idx + 1) % len(KSL_LABELS)
                else:
                    print("  ⚠️  Khong co ban tay hop le de luu.")

            # Xoá mẫu cuối (D)
            elif key in (ord("d"), ord("D")):
                cur_label = KSL_LABELS[label_idx]
                if counts.get(cur_label, 0) > 0:
                    # Đọc lại CSV, bỏ dòng cuối của label này
                    with open(csv_path, "r", encoding="utf-8", newline="") as f:
                        rows = list(csv.reader(f))
                    # Tìm dòng cuối của cur_label (cột 1 = label)
                    header = rows[0]
                    data   = rows[1:]
                    label_col = header.index("label")
                    # Xoá dòng cuối của label này
                    for i in range(len(data) - 1, -1, -1):
                        if data[i][label_col] == cur_label:
                            data.pop(i)
                            break
                    # Ghi lại
                    csv_file.close()
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer2 = csv.writer(f)
                        writer2.writerow(header)
                        writer2.writerows(data)
                    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
                    writer   = csv.writer(csv_file)
                    counts[cur_label] -= 1
                    print(f"  🗑  Xoa 1 mau {cur_label}. Con lai: {counts[cur_label]}")
                else:
                    print(f"  ⚠️  Khong co mau nao cua {cur_label} de xoa.")

            # Chuyển ký hiệu ← (A hoặc phím trái)
            elif key in (ord("a"), ord("A"), 81, 2):  # 81=left arrow msmf, 2=left dshow
                label_idx = (label_idx - 1) % len(KSL_LABELS)
                print(f"  ← Chuyen sang: {KSL_LABELS[label_idx]} ({KSL_NAMES[label_idx]})")

            # Chuyển ký hiệu → (F hoặc phím phải)
            elif key in (ord("f"), ord("F"), 83, 3):  # 83=right arrow
                label_idx = (label_idx + 1) % len(KSL_LABELS)
                print(f"  → Chuyen sang: {KSL_LABELS[label_idx]} ({KSL_NAMES[label_idx]})")

            # Chọn nhanh bằng số 1-9, 0
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(KSL_LABELS):
                    label_idx = idx
                    print(f"  Jump → {KSL_LABELS[label_idx]} ({KSL_NAMES[label_idx]})")
            elif key == ord("0"):
                idx = 9
                if idx < len(KSL_LABELS):
                    label_idx = idx
                    print(f"  Jump → {KSL_LABELS[label_idx]} ({KSL_NAMES[label_idx]})")

            # Kiểm tra cửa sổ bị đóng
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        cap.release()
        detector.close()
        csv_file.close()
        cv2.destroyAllWindows()
        total = sum(counts.values())
        print(f"\n{'='*60}")
        print(f"  Ket qua thu thap:")
        for lbl, name in zip(KSL_LABELS, KSL_NAMES):
            n   = counts.get(lbl, 0)
            bar = "█" * min(n * 20 // max(args.target, 1), 20)
            pct = f"{n}/{args.target}"
            print(f"    {lbl} ({name:<8}) {bar:<20} {pct}")
        print(f"\n  Tong: {total} mau  |  Output: {csv_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
