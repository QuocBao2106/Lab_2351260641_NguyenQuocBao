
from pathlib import Path
import csv

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
RESULT_DIR = Path("results")

N_STATES = 5
N_ITER = 100

COVARIANCE_TYPES = ["diag", "full"]


# ==========================================
# 2. TAO HMM TRAI SANG PHAI
# ==========================================

def make_lr_hmm(covariance_type):

    start = np.zeros(N_STATES)
    start[0] = 1.0

    trans = np.zeros((N_STATES, N_STATES))

    for i in range(N_STATES - 1):
        trans[i, i] = 0.7
        trans[i, i + 1] = 0.3

    trans[-1, -1] = 1.0

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type=covariance_type,
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
# 3. DOC DU LIEU DUNG CHO HUAN LUYEN
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

        # Dung day du 39 chieu MFCC dong
        F = np.load(path, allow_pickle=False)

        if F.ndim != 2 or F.shape[1] != 39:
            raise ValueError(
                f"Shape khong hop le: {path}, {F.shape}"
            )

        if not np.isfinite(F).all():
            raise ValueError(
                f"Phat hien NaN/inf trong {path}"
            )

        features.append(F)
        lengths.append(len(F))

    return np.vstack(features), lengths


# ==========================================
# 4. HUAN LUYEN CAC MO HINH
# ==========================================

def train_models(covariance_type):

    models = {}

    print(f"\n=== COVARIANCE: {covariance_type.upper()} ===")

    for label in LABELS:

        X, lengths = load_train_data(label)

        model = make_lr_hmm(covariance_type)

        model.fit(X, lengths=lengths)

        # Kiem tra ma tran chuyen trang thai
        trans = model.transmat_

        if not np.isfinite(trans).all():
            raise ValueError(
                f"Transmat co NaN/inf: {label}"
            )

        if not np.allclose(trans.sum(axis=1), 1.0):
            raise ValueError(
                f"Tong hang transmat khong bang 1: {label}"
            )

        # Kiem tra cac tham so Gaussian
        if not np.isfinite(model.means_).all():
            raise ValueError(
                f"Means khong hop le: {label}"
            )

        if not np.isfinite(model.covars_).all():
            raise ValueError(
                f"Covariance khong hop le: {label}"
            )

        models[label] = model

        print(
            f"{label}: "
            f"converged={model.monitor_.converged}, "
            f"iterations={model.monitor_.iter}"
        )

    return models


# ==========================================
# 5. DANH GIA TREN TAP TEST
# ==========================================

def evaluate_models(models):

    y_true = []
    y_pred = []

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

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Shape khong hop le: {path}"
                )

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

            print(
                f"{path.name}: "
                f"true={true_label}, pred={prediction}"
            )

    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    return accuracy, cm


# ==========================================
# 6. CHAY SO SANH
# ==========================================

def main():

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    print("=== SO SANH DIAG VA FULL COVARIANCE ===")
    print("Su dung MFCC dong 39 chieu va 5 states.")

    for covariance_type in COVARIANCE_TYPES:

        try:
            models = train_models(covariance_type)

            accuracy, cm = evaluate_models(models)

            print(
                f"\n========== {covariance_type.upper()} =========="
            )

            print(f"Accuracy: {accuracy * 100:.2f}%")
            print("Confusion matrix:")
            print(cm)

            summary.append({
                "covariance_type": covariance_type,
                "accuracy": accuracy,
                "correct_files": int(round(accuracy * 10)),
                "total_test_files": 10,
                "status": "success"
            })

        except (ValueError, np.linalg.LinAlgError) as error:

            print(
                f"\nCAU HINH {covariance_type.upper()} "
                "KHONG CHAY DUOC."
            )
            print("Chi tiet:", error)

            summary.append({
                "covariance_type": covariance_type,
                "accuracy": "",
                "correct_files": "",
                "total_test_files": 10,
                "status": f"failed: {error}"
            })

    # Luu bang tong ket
    output_path = RESULT_DIR / "covariance_comparison.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "covariance_type",
                "accuracy",
                "correct_files",
                "total_test_files",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\n========== TONG KET COVARIANCE ==========")

    for item in summary:

        if item["status"] == "success":
            print(
                f"{item['covariance_type']}: "
                f"{item['accuracy'] * 100:.2f}% "
                f"({item['correct_files']}/10 file)"
            )
        else:
            print(
                f"{item['covariance_type']}: "
                f"{item['status']}"
            )

    print("\nDa luu:", output_path)
    print("DA HOAN THANH THI NGHIEM COVARIANCE!")


if __name__ == "__main__":
    main()
