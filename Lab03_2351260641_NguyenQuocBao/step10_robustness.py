
from pathlib import Path
import csv

import numpy as np
import librosa
import soundfile as sf
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix

# Dung lai ham trich dac trung da kiem tra o Buoc 4
from step4_extract_features import extract_feat


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

SAMPLE_RATE = 16000
SNR_DB = 15

DATASET_DIR = Path("dataset")
FEATURE_DIR = Path("features")
MODEL_PATH = Path("models") / "hmm_models.joblib"
RESULT_DIR = Path("results")

NOISY_DIR = RESULT_DIR / "noisy_test_wav"


# ==========================================
# 2. THEM NHIỄU GAUSSIAN
# ==========================================

def add_noise(y, snr_db, rng):

    # Cong suat tin hieu goc
    signal_power = np.mean(y ** 2)

    if signal_power < 1e-12:
        raise ValueError(
            "Tin hieu qua nho hoac gan nhu im lang."
        )

    # Tao nhieu Gaussian
    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(y)
    ).astype(np.float32)

    noise_power = np.mean(noise ** 2)

    # Dieu chinh muc nhieu theo SNR
    target_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise = noise * np.sqrt(
        target_noise_power / noise_power
    )

    noisy = y + noise

    # Tranh clipping khi luu WAV
    peak = np.max(np.abs(noisy))

    if peak > 0.99:
        noisy = noisy * (0.99 / peak)

    return noisy.astype(np.float32)


# ==========================================
# 3. NHAN DIEN MOT MA TRAN DAC TRUNG
# ==========================================

def predict_hmm(F, models):

    scores = {}

    for label in LABELS:
        score = float(models[label].score(F))

        if not np.isfinite(score):
            raise ValueError(
                f"Likelihood khong hop le: {label}"
            )

        scores[label] = score

    return max(scores, key=scores.get)


# ==========================================
# 4. CHAY THI NGHIEM
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    models = joblib.load(MODEL_PATH)

    NOISY_DIR.mkdir(parents=True, exist_ok=True)

    # Seed co dinh de co the lap lai thi nghiem
    rng = np.random.default_rng(42)

    y_true = []
    pred_clean = []
    pred_noisy = []
    records = []

    print("=== THI NGHIEM ROBUSTNESS ===")
    print("So file test du kien: 10")
    print("SNR:", SNR_DB, "dB")

    for label in LABELS:

        wav_dir = DATASET_DIR / "test" / label
        feature_dir = FEATURE_DIR / "test" / label

        wav_files = sorted(wav_dir.glob("*.wav"))

        if len(wav_files) != 2:
            raise ValueError(
                f"Lop {label} can 2 file WAV test, "
                f"nhung tim thay {len(wav_files)}."
            )

        for wav_path in wav_files:

            feature_path = (
                feature_dir / f"{wav_path.stem}.npy"
            )

            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Khong tim thay feature: {feature_path}"
                )

            # ----------------------------------
            # A. NHAN DIEN FILE SACH
            # ----------------------------------

            F_clean = np.load(
                feature_path,
                allow_pickle=False
            )

            clean_prediction = predict_hmm(
                F_clean, models
            )

            # ----------------------------------
            # B. TAO FILE CO NHIỄU
            # ----------------------------------

            y, sr = librosa.load(
                str(wav_path),
                sr=SAMPLE_RATE,
                mono=True
            )

            noisy_audio = add_noise(
                y, SNR_DB, rng
            )

            noisy_wav_dir = NOISY_DIR / label
            noisy_wav_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            noisy_path = (
                noisy_wav_dir
                / f"{wav_path.stem}_noise.wav"
            )

            sf.write(
                str(noisy_path),
                noisy_audio,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            # ----------------------------------
            # C. TRICH DAC TRUNG FILE CO NHIỄU
            # ----------------------------------

            F_noisy = extract_feat(noisy_path)

            if (
                F_noisy.ndim != 2
                or F_noisy.shape[1] != 39
            ):
                raise ValueError(
                    f"Feature co nhieu khong hop le: "
                    f"{F_noisy.shape}"
                )

            # ----------------------------------
            # D. NHAN DIEN FILE CO NHIỄU
            # ----------------------------------

            noisy_prediction = predict_hmm(
                F_noisy, models
            )

            y_true.append(label)
            pred_clean.append(clean_prediction)
            pred_noisy.append(noisy_prediction)

            records.append({
                "file": wav_path.name,
                "true_label": label,
                "clean_prediction": clean_prediction,
                "noisy_prediction": noisy_prediction,
                "snr_db": SNR_DB
            })

            print(f"\nFile: {wav_path.name}")
            print("Nhan that:", label)
            print("Du doan sach:", clean_prediction)
            print("Du doan co nhieu:", noisy_prediction)

    # ==========================================
    # 5. TINH ACCURACY VA CONFUSION MATRIX
    # ==========================================

    acc_clean = accuracy_score(
        y_true, pred_clean
    )

    acc_noisy = accuracy_score(
        y_true, pred_noisy
    )

    cm_clean = confusion_matrix(
        y_true, pred_clean, labels=LABELS
    )

    cm_noisy = confusion_matrix(
        y_true, pred_noisy, labels=LABELS
    )

    print("\n========== KET QUA ROBUSTNESS ==========")

    print(f"Accuracy du lieu sach: {acc_clean * 100:.2f}%")
    print(f"Accuracy khi co nhieu: {acc_noisy * 100:.2f}%")

    print(
        "Muc thay doi accuracy:",
        f"{(acc_noisy - acc_clean) * 100:.2f}",
        "diem phan tram"
    )

    print("\nConfusion matrix - du lieu sach:")
    print(cm_clean)

    print("\nConfusion matrix - du lieu co nhieu:")
    print(cm_noisy)

    # ==========================================
    # 6. LUU KET QUA
    # ==========================================

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / "robustness_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nDa luu ket qua:", csv_path)
    print("Da luu WAV co nhieu tai:", NOISY_DIR)
    print("\nDA HOAN THANH THI NGHIEM ROBUSTNESS!")


if __name__ == "__main__":
    main()
