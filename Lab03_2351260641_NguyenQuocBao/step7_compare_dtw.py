
from pathlib import Path
import csv
import time

import joblib
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
MODEL_PATH = Path("models") / "hmm_models.joblib"
RESULT_DIR = Path("results")


# ==========================================
# 2. TINH KHOANG CACH DTW
# ==========================================

def dtw_distance(A, B):
    """
    A, B: hai ma tran dac trung co 39 cot.

    Tra ve khoang cach DTW duoc chuan hoa
    theo so buoc cua duong di.
    """

    # Khoang cach Euclidean giua moi cap frame
    local_cost = cdist(A, B, metric="euclidean")

    n, m = local_cost.shape

    # Bang chi phi tich luy
    cost = np.full((n + 1, m + 1), np.inf)

    # So buoc cua duong di
    length = np.zeros((n + 1, m + 1), dtype=np.int32)

    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            # Ba huong di co the chon
            candidates = [
                cost[i - 1, j - 1],  # Cheo
                cost[i - 1, j],      # Tu tren xuong
                cost[i, j - 1]       # Tu trai sang
            ]

            best = int(np.argmin(candidates))

            if best == 0:
                prev_cost = cost[i - 1, j - 1]
                prev_length = length[i - 1, j - 1]

            elif best == 1:
                prev_cost = cost[i - 1, j]
                prev_length = length[i - 1, j]

            else:
                prev_cost = cost[i, j - 1]
                prev_length = length[i, j - 1]

            cost[i, j] = (
                local_cost[i - 1, j - 1] + prev_cost
            )

            length[i, j] = prev_length + 1

    # Chuan hoa de de so sanh cac chuoi dai ngan khac nhau
    return cost[n, m] / max(length[n, m], 1)


# ==========================================
# 3. DU DOAN BANG DTW
# ==========================================

def recognize_dtw(F, templates):

    # Diem nho hon la tot hon
    distances = {}

    for label in LABELS:

        template_distances = []

        for template_name, template in templates[label]:
            distance = dtw_distance(F, template)
            template_distances.append(distance)

        # Dung khoang cach nho nhat trong 6 mau cua lop
        distances[label] = min(template_distances)

    prediction = min(distances, key=distances.get)

    return prediction, distances


# ==========================================
# 4. VE CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    output_path
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return cm


# ==========================================
# 5. CHAY THI NGHIEM
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Nap cac HMM da huan luyen o Buoc 5
    models = joblib.load(MODEL_PATH)

    # Nap 30 mau train lam template cho DTW
    templates = {}

    print("=== CHUAN BI TEMPLATE DTW ===")

    template_start = time.perf_counter()

    for label in LABELS:

        train_dir = FEATURE_DIR / "train" / label
        files = sorted(train_dir.glob("*.npy"))

        if len(files) != 6:
            raise ValueError(
                f"Lop {label} can 6 file train, "
                f"nhung tim thay {len(files)}."
            )

        templates[label] = []

        for path in files:

            F = np.load(path, allow_pickle=False)

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Shape khong hop le: {path}, {F.shape}"
                )

            templates[label].append((path.name, F))

    template_time = time.perf_counter() - template_start

    print("Tong so template:", sum(
        len(templates[label]) for label in LABELS
    ))
    print(f"Thoi gian nap template: {template_time:.4f} giay")

    # Cac bien luu ket qua
    y_true = []
    y_hmm = []
    y_dtw = []
    records = []

    hmm_times = []
    dtw_times = []

    print("\n=== BAT DAU SO SANH HMM VA DTW ===")

    # Duyet cung tap test cho ca hai phuong phap
    for true_label in LABELS:

        test_dir = FEATURE_DIR / "test" / true_label
        test_files = sorted(test_dir.glob("*.npy"))

        if len(test_files) != 2:
            raise ValueError(
                f"Lop {true_label} can 2 file test, "
                f"nhung tim thay {len(test_files)}."
            )

        for path in test_files:

            F = np.load(path, allow_pickle=False)

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Shape khong hop le: {path}, {F.shape}"
                )

            # ----- HMM -----
            start = time.perf_counter()

            hmm_scores = {
                label: float(models[label].score(F))
                for label in LABELS
            }

            if not all(
                np.isfinite(score)
                for score in hmm_scores.values()
            ):
                raise ValueError(
                    f"HMM score khong hop le: {path.name}"
                )

            hmm_pred = max(
                hmm_scores,
                key=hmm_scores.get
            )

            hmm_elapsed = time.perf_counter() - start
            hmm_times.append(hmm_elapsed)

            # ----- DTW -----
            start = time.perf_counter()

            dtw_pred, dtw_distances = recognize_dtw(
                F, templates
            )

            dtw_elapsed = time.perf_counter() - start
            dtw_times.append(dtw_elapsed)

            # Luu ket qua
            y_true.append(true_label)
            y_hmm.append(hmm_pred)
            y_dtw.append(dtw_pred)

            record = {
                "file": path.name,
                "true_label": true_label,
                "hmm_prediction": hmm_pred,
                "dtw_prediction": dtw_pred,
                "hmm_time_ms": hmm_elapsed * 1000,
                "dtw_time_ms": dtw_elapsed * 1000
            }

            for label in LABELS:
                record[f"hmm_score_{label}"] = hmm_scores[label]
                record[f"dtw_distance_{label}"] = dtw_distances[label]

            records.append(record)

            print(f"\nFile: {path.name}")
            print("Nhan that:", true_label)
            print("HMM du doan:", hmm_pred)
            print("DTW du doan:", dtw_pred)

            if hmm_pred != true_label or dtw_pred != true_label:
                print("CAN PHAN TICH: Co it nhat mot phuong phap nhan sai.")

    # ======================================
    # 6. SO SANH ACCURACY VA THOI GIAN
    # ======================================

    hmm_accuracy = accuracy_score(y_true, y_hmm)
    dtw_accuracy = accuracy_score(y_true, y_dtw)

    print("\n========== KET QUA SO SANH ==========")
    print(f"HMM Accuracy: {hmm_accuracy * 100:.2f}%")
    print(f"DTW Accuracy: {dtw_accuracy * 100:.2f}%")

    print(
        "HMM thoi gian nhan dang trung binh: "
        f"{np.mean(hmm_times) * 1000:.3f} ms/file"
    )

    print(
        "DTW thoi gian nhan dang trung binh: "
        f"{np.mean(dtw_times) * 1000:.3f} ms/file"
    )

    # ======================================
    # 7. VE HAI CONFUSION MATRIX
    # ======================================

    cm_hmm = save_confusion_matrix(
        y_true,
        y_hmm,
        "HMM Confusion Matrix",
        RESULT_DIR / "comparison_hmm_cm.png"
    )

    cm_dtw = save_confusion_matrix(
        y_true,
        y_dtw,
        "DTW Confusion Matrix",
        RESULT_DIR / "comparison_dtw_cm.png"
    )

    print("\nConfusion matrix HMM:")
    print(cm_hmm)

    print("\nConfusion matrix DTW:")
    print(cm_dtw)

    # ======================================
    # 8. LUU BANG SO SANH CSV
    # ======================================

    csv_path = RESULT_DIR / "comparison_hmm_dtw.csv"

    fieldnames = list(records[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nDa luu bang so sanh:", csv_path)
    print("Da luu:", RESULT_DIR / "comparison_hmm_cm.png")
    print("Da luu:", RESULT_DIR / "comparison_dtw_cm.png")
    print("\nDA HOAN THANH SO SANH HMM VA DTW!")


if __name__ == "__main__":
    main()
