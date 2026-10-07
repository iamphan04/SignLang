"""
Merge nhiều file CSV từ nhiều người thành 1 file dataset tổng hợp.
Kiểm tra tính hợp lệ: đúng 42 features, labels đúng danh sách VSL.

Chạy: python merge_vsl_csv.py
Output: data/vsl_collected/vsl_merged.csv + stats
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "vsl_collected"

VSL_LABELS = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]
FEATURE_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]
EXPECTED_HDR = ["person_id", "label", "label_name", "session", "timestamp"] + FEATURE_COLS


def validate_row(row: dict, source_file: str, line: int) -> str | None:
    """Trả về None nếu hợp lệ, hoặc chuỗi mô tả lỗi."""
    lbl = row.get("label", "")
    if lbl not in VSL_LABELS:
        return f"{source_file}:{line} — label không hợp lệ: '{lbl}'"
    for col in FEATURE_COLS:
        val = row.get(col, "")
        try:
            float(val)
        except ValueError:
            return f"{source_file}:{line} — feature '{col}' không phải số: '{val}'"
    return None


def main():
    csv_files = sorted(DATA_DIR.glob("vsl_*.csv"))
    # Bỏ file merged nếu đã tồn tại trong danh sách input
    csv_files = [f for f in csv_files if f.stem != "vsl_merged"]

    if not csv_files:
        print(f"❌ Không tìm thấy file CSV nào trong {DATA_DIR}")
        print("   Thu thập data trước bằng collect_vsl.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  VSL CSV Merger")
    print(f"  Tìm thấy {len(csv_files)} file(s):")
    for f in csv_files:
        print(f"    - {f.name}")
    print(f"{'='*60}\n")

    all_rows   = []
    errors     = []
    per_file   = {}
    per_person = defaultdict(lambda: defaultdict(int))

    for csv_path in csv_files:
        file_rows = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Kiểm tra header
            if reader.fieldnames != EXPECTED_HDR:
                print(f"⚠️  Header sai trong {csv_path.name}. Bỏ qua file này.")
                print(f"   Mong đợi: {EXPECTED_HDR[:6]}...")
                print(f"   Thực tế:  {reader.fieldnames[:6] if reader.fieldnames else '(empty)'}...")
                continue

            for line_num, row in enumerate(reader, start=2):
                err = validate_row(row, csv_path.name, line_num)
                if err:
                    errors.append(err)
                    continue
                all_rows.append(row)
                file_rows += 1
                person = row.get("person_id", "unknown")
                lbl    = row["label"]
                per_person[person][lbl] += 1

        per_file[csv_path.name] = file_rows
        print(f"  ✅ {csv_path.name}: {file_rows} dòng hợp lệ")

    if errors:
        print(f"\n⚠️  {len(errors)} dòng bị lỗi (đã bỏ qua):")
        for e in errors[:10]:
            print(f"   {e}")
        if len(errors) > 10:
            print(f"   ... và {len(errors)-10} lỗi khác")

    if not all_rows:
        print("\n❌ Không có dòng hợp lệ nào để merge.")
        sys.exit(1)

    # Ghi file merged
    merged_path = DATA_DIR / "vsl_merged.csv"
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_HDR)
        writer.writeheader()
        writer.writerows(all_rows)

    # In thống kê
    print(f"\n{'='*60}")
    print(f"  Thống kê dataset:")
    print(f"  Tổng mẫu: {len(all_rows)}")
    print(f"  Số người: {len(per_person)}")
    print(f"\n  Phân bố theo label:")
    for lbl in VSL_LABELS:
        total = sum(p.get(lbl, 0) for p in per_person.values())
        by_person = "  ".join(
            f"{pid}:{cnt.get(lbl,0)}"
            for pid, cnt in sorted(per_person.items())
        )
        bar = "█" * min(total // 10, 25)
        print(f"    {lbl}  {total:>4}  {bar:<25}  [{by_person}]")

    print(f"\n  Phân bố theo người:")
    for person, label_counts in sorted(per_person.items()):
        total_p = sum(label_counts.values())
        print(f"    {person}: {total_p} mẫu ({len(label_counts)} ký hiệu)")

    missing = [lbl for lbl in VSL_LABELS
               if sum(p.get(lbl, 0) for p in per_person.values()) == 0]
    if missing:
        print(f"\n  ⚠️  Ký hiệu chưa có dữ liệu: {missing}")

    low = [lbl for lbl in VSL_LABELS
           if sum(p.get(lbl, 0) for p in per_person.values()) < 50]
    if low:
        print(f"  ⚠️  Ký hiệu ít mẫu (<50): {low}")

    print(f"\n  Output: {merged_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
