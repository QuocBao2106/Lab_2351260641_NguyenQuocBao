from pathlib import Path
import csv
import time

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]
STATE_COUNTS = [3, 5, 7]

FEATURE_DIR = Path("features")
RESULT_DIR = Path("results")

N_ITER = 100


# ==========================================
# 2. TAO HMM TRAI SANG PHAI
# ==========================================

def make_lr_hmm(n_states):

    start = np.zeros(n_states)
    start[0] = 1.0

    trans = np.zeros((n_states, n_states))

    for i in range(n_states - 1):
        trans[i, i] = 0.7
        trans[i, i + 1] = 0.3

    # Trang thai cuoi chi tu lap
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
# 3. DOC DU LIEU HUAN LUYEN
# ==========================================

def load_train_data(label):

    folder = FEATURE_DIR / "train" / label
    files = sorted(folder.glob("*.npy"))

    if len(files) != 6:
        raise ValueError(
            f"Lop {label} can 6 file train, "
            f"nhung tim thay {len(files)}."
        )

    features = []
    lengths = []

    for path in files:

        F = np.load(path, allow_pickle=False)

        if F.ndim != 2 or F.shape[1] != 39:
            raise ValueError(
                f"Shape khong hop le: {path}: {F.shape}"
            )

        if not np.isfinite(F).all():
            raise ValueError(
                f"Du lieu co NaN/inf: {path}"
            )

        features.append(F)
        lengths.append(len(F))

    return np.vstack(features), lengths


# ==========================================
# 4. HUAN LUYEN CAC MO HINH
# ==========================================

def train_models(n_states):

    models = {}

    print(f"\n--- HUAN LUYEN VOI {n_states} STATES ---")

    for label in LABELS:

        X, lengths = load_train_data(label)

        model = make_lr_hmm(n_states)

        start_time = time.perf_counter()
        model.fit(X, lengths=lengths)
        elapsed = time.perf_counter() - start_time

        # Kiem tra tham so sau huan luyen
        trans = model.transmat_

        if not np.isfinite(trans).all():
            raise ValueError(
                f"Transmat khong hop le: {label}, "
                f"{n_states} states"
            )

        if not np.allclose(trans.sum(axis=1), 1.0):
            raise ValueError(
                f"Tong hang transmat khong bang 1: {label}"
            )

        models[label] = model

        print(
            f"{label}: "
            f"converged={model.monitor_.converged}, "
            f"iterations={model.monitor_.iter}, "
            f"train_time={elapsed:.3f}s"
        )

    return models


# ==========================================
# 5. DANH GIA TREN 10 FILE TEST
# ==========================================

def evaluate_models(models, n_states):

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

            scores = {
                label: float(models[label].score(F))
                for label in LABELS
            }

            if not all(
                np.isfinite(score)
                for score in scores.values()
            ):
                raise ValueError(
                    f"Likelihood khong hop le: {path.name}"
                )

            prediction = max(scores, key=scores.get)

            y_true.append(true_label)
            y_pred.append(prediction)

            records.append({
                "n_states": n_states,
                "file": path.name,
                "true_label": true_label,
                "prediction": prediction,
                "correct": int(prediction == true_label)
            })

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

    print("=== THI NGHIEM SO TRANG THAI HMM ===")

    for n_states in STATE_COUNTS:

        models = train_models(n_states)

        accuracy, cm, records = evaluate_models(
            models, n_states
        )

        all_records.extend(records)

        print(f"\n========== STATES = {n_states} ==========")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion matrix:")
        print(cm)

    # Luu ket qua tung file cua ca ba cau hinh
    output_path = RESULT_DIR / "state_count_results.csv"

    fieldnames = [
        "n_states",
        "file",
        "true_label",
        "prediction",
        "correct"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\nDa luu chi tiet:", output_path)
    print("\nDA HOAN THANH THI NGHIEM SO STATES!")


if __name__ == "__main__":
    main()