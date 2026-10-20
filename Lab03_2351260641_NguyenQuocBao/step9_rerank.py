
from pathlib import Path
from collections import Counter
import math

import joblib
import numpy as np


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
MODEL_PATH = Path("models") / "hmm_models.joblib"
CORPUS_PATH = Path("corpus_bigram.txt")

K = 0.5

# Tu dung truoc da biet, dung de minh hoa
PREVIOUS_WORD = "mo"

# Cac trong so can thu nghiem
LM_WEIGHTS = [0, 0.1, 0.3, 1.0]


# ==========================================
# 2. DEM BIGRAM TU CORPUS
# ==========================================

def load_bigram_counts(path):

    bigram_counts = Counter()
    context_counts = Counter()

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip().lower()

            if not line or line.startswith("#"):
                continue

            words = line.split()

            for word in words:
                if word not in LABELS:
                    raise ValueError(
                        f"Tu khong hop le trong corpus: {word}"
                    )

            # Dem cac cap tu lien tiep
            for i in range(len(words) - 1):

                previous = words[i]
                current = words[i + 1]

                bigram_counts[(previous, current)] += 1
                context_counts[previous] += 1

    return bigram_counts, context_counts


# ==========================================
# 3. TINH XAC SUAT BIGRAM ADD-K
# ==========================================

def bigram_probability(
    previous,
    current,
    bigram_counts,
    context_counts
):

    V = len(LABELS)

    numerator = (
        bigram_counts[(previous, current)] + K
    )

    denominator = (
        context_counts[previous] + K * V
    )

    return numerator / denominator


# ==========================================
# 4. TINH DIEM AM HOC BANG HMM
# ==========================================

def acoustic_scores(F, models):

    T = len(F)

    scores = {}

    for label in LABELS:

        # Chuan hoa likelihood theo so frame
        scores[label] = float(
            models[label].score(F) / T
        )

        if not np.isfinite(scores[label]):
            raise ValueError(
                f"Score HMM khong hop le: {label}"
            )

    return scores


# ==========================================
# 5. TIM FILE CO HAI UNG VIEN GAN NHAU
# ==========================================

def find_closest_test(models):

    best_trial = None

    for true_label in LABELS:

        folder = FEATURE_DIR / "test" / true_label

        for path in sorted(folder.glob("*.npy")):

            F = np.load(path, allow_pickle=False)

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Shape khong hop le: {path}, {F.shape}"
                )

            scores = acoustic_scores(F, models)

            ranking = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            # Khoang cach diem giua ung vien dau va thu hai
            margin = ranking[0][1] - ranking[1][1]

            trial = {
                "path": path,
                "true_label": true_label,
                "features": F,
                "scores": scores,
                "margin": margin,
                "ranking": ranking
            }

            if (
                best_trial is None
                or margin < best_trial["margin"]
            ):
                best_trial = trial

    if best_trial is None:
        raise FileNotFoundError(
            "Khong tim thay file test nao."
        )

    return best_trial


# ==========================================
# 6. RE-RANKING KET HOP HMM VA BIGRAM
# ==========================================

def rerank(
    acoustic,
    previous_word,
    bigram_counts,
    context_counts,
    lm_weight
):

    combined_scores = {}

    for word in LABELS:

        probability = bigram_probability(
            previous_word,
            word,
            bigram_counts,
            context_counts
        )

        # Diem ket hop am hoc va ngon ngu
        combined_scores[word] = (
            acoustic[word]
            + lm_weight * math.log(probability)
        )

    ranking = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return combined_scores, ranking


# ==========================================
# 7. CHAY THI NGHIEM
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh: {MODEL_PATH}"
        )

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay corpus: {CORPUS_PATH}"
        )

    # Nap cac mo hinh HMM
    models = joblib.load(MODEL_PATH)

    # Dem bigram tren cung corpus da dung o Buoc 9.1
    bigram_counts, context_counts = load_bigram_counts(
        CORPUS_PATH
    )

    if PREVIOUS_WORD not in LABELS:
        raise ValueError(
            f"Tu dung truoc khong hop le: {PREVIOUS_WORD}"
        )

    # Tim file test co top-1 va top-2 am hoc gan nhau
    trial = find_closest_test(models)

    path = trial["path"]
    true_label = trial["true_label"]
    acoustic = trial["scores"]

    print("========== FILE THU NGHIEM ==========")
    print("File:", path.name)
    print("Nhan that:", true_label)
    print("Tu dung truoc gia dinh:", PREVIOUS_WORD)
    print("So frame:", len(trial["features"]))

    print("\nDiem am hoc HMM (log-likelihood/frame):")

    for word, score in trial["ranking"]:
        print(f"{word:6s}: {score:.6f}")

    print(
        "\nKhoang cach Top-1 va Top-2:",
        f"{trial['margin']:.6f}"
    )

    # Tinh diem ngon ngu va re-rank voi tung lambda
    print("\n========== THU CAC LAMBDA_LM ==========")

    results = []

    for weight in LM_WEIGHTS:

        combined, ranking = rerank(
            acoustic,
            PREVIOUS_WORD,
            bigram_counts,
            context_counts,
            weight
        )

        predicted = ranking[0][0]

        results.append((weight, predicted))

        print(f"\nlambda_LM = {weight}")

        for word, score in ranking:

            probability = bigram_probability(
                PREVIOUS_WORD,
                word,
                bigram_counts,
                context_counts
            )

            print(
                f"{word:6s}: "
                f"combined_score = {score:.6f}, "
                f"P_bigram = {probability:.4f}"
            )

        print("Nhan du doan sau re-ranking:", predicted)

        if predicted == true_label:
            print("Ket qua: NHAN DUNG")
        else:
            print("Ket qua: NHAN SAI")

    # Tong hop ket qua
    print("\n========== TONG HOP ==========")

    baseline_prediction = results[0][1]

    for weight, predicted in results:

        if weight == 0:
            description = "Chi dung HMM"
        elif predicted == true_label and baseline_prediction != true_label:
            description = "Bigram da giup sua loi"
        elif predicted != true_label and baseline_prediction == true_label:
            description = "Bigram lam sai ket qua HMM"
        else:
            description = "Chua thay doi tinh dung/sai so voi HMM"

        print(
            f"lambda={weight:<3} | "
            f"du doan={predicted:<6} | "
            f"{description}"
        )

    print("\nDA HOAN THANH BUOC 9.2!")


if __name__ == "__main__":
    main()
