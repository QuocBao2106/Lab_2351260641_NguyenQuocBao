
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
RESULT_DIR = Path("results")

N_STATES = 5
N_ITER = 100

FEATURE_TYPES = ["static", "dynamic"]


# ==========================================
# 2. CHON DAC TRUNG
# ==========================================

def select_features(F, feature_type):

    if F.ndim != 2 or F.shape[1] != 39:
        raise ValueError(
            f"Feature goc phai co 39 cot, shape={F.shape}"
        )

    if not np.isfinite(F).all():
        raise ValueError("Du lieu co NaN hoac inf.")

    if feature_type == "static":
        # Chi lay 13 MFCC dau tien
        return F[:, :13]

    elif feature_type == "dynamic":
        # MFCC + Delta + Delta-Delta
        return F

    else:
        raise ValueError(
            f"Loai dac trung khong hop le: {feature_type}"
        )


# ==========================================
# 3. TAO HMM TRAI SANG PHAI
# ==========================================

def make_lr_hmm(n_states=5):

    start = np.zeros(n_states)
    start[0] = 1.0

    trans = np.zeros((n_states, n_states))

    for i in range(n_states - 1):
        trans[i, i] = 0.7
        trans[i, i + 1] = 0.3

    trans[-1, -1] = 1.0

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=1e-3,
        min_covar=1e-3,
        init_params="mc",
        params="mc",
        random_state=42
    )

    model.startprob_ = start
    model.transmat_ = trans

    return model


# ==========================================
# 4. HUAN LUYEN 5 MO HINH
# ==========================================

def train_models(feature_type):

    models = {}

    print(f"\n=== HUAN LUYEN: {feature_type.upper()} ===")

    for label in LABELS:

        folder = FEATURE_DIR / "train" / label
        files = sorted(folder.glob("*.npy"))

        if len(files) != 6:
            raise ValueError(
                f"Lop {label} can 6 file train, "
                f"nhung tim thay {len(files)}."
            )

        feature_list = []
        lengths = []

        for path in files:

            F = np.load(path, allow_pickle=False)

            F = select_features(F, feature_type)

            feature_list.append(F)
            lengths.append(len(F))

        X = np.vstack(feature_list)

        model = make_lr_hmm(N_STATES)
        model.fit(X, lengths=lengths)

        # Kiem tra tham so chuyen trang thai
        trans = model.transmat_

        if not np.isfinite(trans).all():
            raise ValueError(
                f"Transmat co NaN/inf: {label}"
            )

        if not np.allclose(trans.sum(axis=1), 1.0):
            raise ValueError(
                f"Tong hang transmat khong bang 1: {label}"
            )

        models[label] = model

        print(
            f"{label}: feature_shape={X.shape}, "
            f"converged={model.monitor_.converged}"
        )

    return models


# ==========================================
# 5. DANH GIA TAP TEST
# ==========================================

def evaluate_models(models, feature_type):

    y_true = []
    y_pred = []
    records = []

    for true_label in LABELS:

        folder = FEATURE_DIR / "test" / true_label
        files = sorted(folder.glob("*.npy"))

        if len(files) != 2:
            raise ValueError(
                f"Lop {true_label} can 2 file test, "
                f"nhung tim thay {len(files)}."
            )

        for path in files:

            F = np.load(path, allow_pickle=False)

            F = select_features(F, feature_type)

            scores = {
                label: float(models[label].score(F))
                for label in LABELS
            }

            if not all(
                np.isfinite(score)
                for score in scores.values()
            ):
                raise ValueError(
                    f"Score khong hop le: {path.name}"
                )

            prediction = max(scores, key=scores.get)

            y_true.append(true_label)
            y_pred.append(prediction)

            records.append({
                "feature_type": feature_type,
                "file": path.name,
                "true_label": true_label,
                "prediction": prediction,
                "correct": int(prediction == true_label)
            })

            print(
                f"{path.name}: "
                f"true={true_label}, "
                f"pred={prediction}"
            )

    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    return accuracy, cm, records


# ==========================================
# 6. CHAY THI NGHIEM
# ==========================================

def main():

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary = []

    print("=== SO SANH MFCC TINH VA MFCC DONG ===")

    for feature_type in FEATURE_TYPES:

        models = train_models(feature_type)

        accuracy, cm, records = evaluate_models(
            models,
            feature_type
        )

        all_records.extend(records)

        summary.append({
            "feature_type": feature_type,
            "feature_dimension": (
                13 if feature_type == "static" else 39
            ),
            "accuracy": accuracy,
            "correct_files": int(round(accuracy * 10)),
            "total_test_files": 10
        })

        print(f"\n========== {feature_type.upper()} ==========")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("Confusion matrix:")
        print(cm)

        # Luu confusion matrix cho moi loai dac trung
        fig, ax = plt.subplots(figsize=(7, 5))

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

        ax.set_title(f"Confusion Matrix - {feature_type}")
        plt.tight_layout()

        output_image = (
            RESULT_DIR / f"confusion_{feature_type}.png"
        )

        plt.savefig(output_image, dpi=150)
        plt.close()

        print("Da luu:", output_image)

    # ======================================
    # 7. LUU BANG KET QUA
    # ======================================

    summary_path = RESULT_DIR / "feature_comparison.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_type",
                "feature_dimension",
                "accuracy",
                "correct_files",
                "total_test_files"
            ]
        )

        writer.writeheader()
        writer.writerows(summary)

    detail_path = RESULT_DIR / "feature_comparison_details.csv"

    with open(
        detail_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_type",
                "file",
                "true_label",
                "prediction",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\n========== TONG KET ==========")

    for result in summary:
        print(
            f"{result['feature_type']}: "
            f"{result['accuracy'] * 100:.2f}% "
            f"({result['correct_files']}/"
            f"{result['total_test_files']} file)"
        )

    print("\nDa luu:", summary_path)
    print("Da luu:", detail_path)
    print("\nDA HOAN THANH THI NGHIEM DAC TRUNG!")


if __name__ == "__main__":
    main()
