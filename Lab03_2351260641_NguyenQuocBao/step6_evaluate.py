
from pathlib import Path
import csv

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
MODEL_PATH = Path("models") / "hmm_models.joblib"
RESULT_DIR = Path("results")


# ==========================================
# 2. NHAN DIEN MOT FILE
# ==========================================

def recognize(F, models):

    scores = {}

    for label, model in models.items():
        score = float(model.score(F))

        if not np.isfinite(score):
            raise ValueError(
                f"Likelihood khong hop le cho nhan {label}: {score}"
            )

        scores[label] = score

    # Sap xep diem tu cao xuong thap
    ranking = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    prediction = ranking[0][0]

    # Khoang cach giua diem cao nhat va thu hai
    if len(ranking) >= 2:
        margin = ranking[0][1] - ranking[1][1]
    else:
        margin = float("nan")

    return prediction, scores, ranking, margin


# ==========================================
# 3. DANH GIA TREN TAP TEST
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Nap 5 mo hinh da huan luyen
    models = joblib.load(MODEL_PATH)

    print("Cac mo hinh da nap:", list(models.keys()))

    true_labels = []
    predicted_labels = []
    records = []

    print("\n========== NHAN DIEN TAP TEST ==========")

    for true_label in LABELS:

        test_dir = FEATURE_DIR / "test" / true_label
        test_files = sorted(test_dir.glob("*.npy"))

        if not test_files:
            raise FileNotFoundError(
                f"Khong tim thay file test trong {test_dir}"
            )

        for path in test_files:

            # Doc ma tran dac trung T x 39
            F = np.load(path, allow_pickle=False)

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Shape khong hop le: {path}: {F.shape}"
                )

            if not np.isfinite(F).all():
                raise ValueError(
                    f"Du lieu co NaN/inf: {path}"
                )

            prediction, scores, ranking, margin = recognize(
                F, models
            )

            true_labels.append(true_label)
            predicted_labels.append(prediction)

            # Luu ket qua cua tung file
            record = {
                "file": path.name,
                "true_label": true_label,
                "prediction": prediction,
                "top1_minus_top2": margin
            }

            for label in LABELS:
                record[f"score_{label}"] = scores[label]

            records.append(record)

            print("\nFile:", path.name)
            print("Nhan that:", true_label)
            print("Nhan du doan:", prediction)

            print("Likelihood cua 5 mo hinh:")

            for label, score in ranking:
                print(f"  {label:6s}: {score:.4f}")

            print(f"Top-1 minus Top-2: {margin:.4f}")

    # ======================================
    # 4. TINH ACCURACY
    # ======================================

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("\n========== KET QUA DANH GIA ==========")
    print("Tong so file test:", len(true_labels))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ======================================
    # 5. VE CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS
    )

    print("\nConfusion matrix:")
    print(cm)

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

    plt.title("Confusion Matrix - Gaussian HMM")
    plt.tight_layout()

    cm_path = RESULT_DIR / "confusion_matrix_hmm.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("Da luu confusion matrix:", cm_path)

    # ======================================
    # 6. LUU BANG LIKELIHOOD THANH CSV
    # ======================================

    csv_path = RESULT_DIR / "likelihood_scores.csv"

    fieldnames = [
        "file",
        "true_label",
        "prediction",
        "top1_minus_top2"
    ] + [f"score_{label}" for label in LABELS]

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

    print("Da luu bang likelihood:", csv_path)

    # ======================================
    # 7. VE VITERBI PATH CHO 2 FILE TEST
    # ======================================

    selected_labels = ["mo", "dong"]

    for label in selected_labels:

        test_dir = FEATURE_DIR / "test" / label
        test_files = sorted(test_dir.glob("*.npy"))

        if not test_files:
            continue

        path = test_files[0]
        F = np.load(path, allow_pickle=False)

        # Decode bang Viterbi voi model cua nhan that
        logp, states = models[label].decode(
            F,
            algorithm="viterbi"
        )

        plt.figure(figsize=(10, 4))

        plt.step(
            range(len(states)),
            states,
            where="post"
        )

        plt.xlabel("Frame")
        plt.ylabel("State")
        plt.yticks(range(models[label].n_components))
        plt.title(f"Viterbi Alignment - {path.name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        image_path = RESULT_DIR / f"viterbi_{label}.png"

        plt.savefig(image_path, dpi=150)
        plt.close()

        print(
            f"Viterbi {path.name}: "
            f"log-probability = {logp:.4f}"
        )

        print("Da luu:", image_path)

    print("\nDA HOAN THANH BUOC 6!")


if __name__ == "__main__":
    main()
