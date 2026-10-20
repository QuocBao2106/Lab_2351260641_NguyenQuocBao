
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt


# ==========================================
# 1. CAU HINH
# ==========================================

SAMPLE_RATE = 16000

LABELS = ["mo", "dong", "len", "xuong", "dung"]
SPLITS = ["train", "test"]

DATASET_DIR = Path("dataset")
FEATURE_DIR = Path("features")
RESULT_DIR = Path("results")


# ==========================================
# 2. HAM TRICH XUAT DAC TRUNG
# ==========================================

def extract_feat(path):
    # Buoc 1: Doc am thanh, dua ve mono va 16 kHz
    y, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True
    )

    # Buoc 2: Cat khoang im lang
    y, _ = librosa.effects.trim(y, top_db=30)

    # Buoc 3: Tien nhan am thanh
    y = librosa.effects.preemphasis(y, coef=0.97)

    # Buoc 4: Trich 13 MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
        window="hamming",
        n_mels=26,
        fmin=20,
        fmax=sr / 2
    )

    # Buoc 5: Tinh Delta va Delta-Delta
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Buoc 6: Ghep thanh ma tran T x 39
    F = np.vstack([mfcc, delta, delta2]).T

    # Buoc 7: Chuan hoa CMVN theo tung file
    mean = F.mean(axis=0, keepdims=True)
    std = F.std(axis=0, keepdims=True) + 1e-8

    F = (F - mean) / std

    return F


# ==========================================
# 3. KIEM TRA VA XU LY CAC FILE WAV
# ==========================================

def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    preview = {}

    print("=== BAT DAU TRICH XUAT DAC TRUNG ===")

    for split in SPLITS:
        for label in LABELS:

            wav_dir = DATASET_DIR / split / label

            wav_files = sorted(wav_dir.glob("*.wav"))

            if not wav_files:
                print("CANH BAO: Khong co file WAV tai", wav_dir)
                continue

            for wav_path in wav_files:

                # Kiem tra thong tin file goc
                info = sf.info(str(wav_path))

                if info.samplerate != SAMPLE_RATE:
                    print(
                        "CANH BAO TAN SO:",
                        wav_path,
                        info.samplerate
                    )

                if info.channels != 1:
                    print(
                        "CANH BAO SO KENH:",
                        wav_path,
                        info.channels
                    )

                if info.subtype != "PCM_16":
                    print(
                        "CANH BAO DINH DANG:",
                        wav_path,
                        info.subtype
                    )

                # Trich xuat dac trung
                F = extract_feat(wav_path)

                # Kiem tra du lieu dau ra
                if F.ndim != 2 or F.shape[1] != 39:
                    raise ValueError(
                        f"Shape khong hop le: {wav_path}: {F.shape}"
                    )

                if not np.isfinite(F).all():
                    raise ValueError(
                        f"Dac trung co NaN hoac inf: {wav_path}"
                    )

                # Luu dac trung thanh file NPY
                output_dir = FEATURE_DIR / split / label
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / (wav_path.stem + ".npy")
                np.save(output_path, F)

                total_files += 1

                print(
                    f"{wav_path} -> shape = {F.shape}"
                )

                # Luu 2 file train de ve heatmap
                if split == "train" and wav_path.stem in [
                    "mo_01", "dong_01"
                ]:
                    preview[label] = F

    print("\n=== KET QUA ===")
    print("Tong so file da xu ly:", total_files)

    # ======================================
    # 4. VE HEATMAP MFCC, DELTA, DELTA-DELTA
    # ======================================

    selected_labels = ["mo", "dong"]

    if all(label in preview for label in selected_labels):

        fig, axes = plt.subplots(
            3, 2,
            figsize=(14, 9)
        )

        feature_names = [
            "MFCC",
            "Delta MFCC",
            "Delta-Delta MFCC"
        ]

        for col, label in enumerate(selected_labels):

            F = preview[label]

            groups = [
                F[:, 0:13].T,
                F[:, 13:26].T,
                F[:, 26:39].T
            ]

            for row in range(3):

                axes[row, col].imshow(
                    groups[row],
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    cmap="viridis"
                )

                axes[row, col].set_title(
                    f"{feature_names[row]} - {label}"
                )

                axes[row, col].set_xlabel("Frame")
                axes[row, col].set_ylabel("He so")

        plt.tight_layout()

        output_image = RESULT_DIR / "feature_heatmaps.png"
        plt.savefig(output_image, dpi=150)

        plt.show()

        print("\nDa luu heatmap tai:", output_image)

    else:
        print(
            "Chua du file mo_01.wav va dong_01.wav "
            "de ve heatmap."
        )

    print("\nDA HOAN THANH BUOC TRICH XUAT DAC TRUNG!")


if __name__ == "__main__":
    main()
