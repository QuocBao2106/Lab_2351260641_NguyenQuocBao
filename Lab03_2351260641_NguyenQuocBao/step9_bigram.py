
from pathlib import Path
from collections import Counter

# ==========================================
# 1. CAU HINH
# ==========================================

CORPUS_PATH = Path("corpus_bigram.txt")

LABELS = ["mo", "dong", "len", "xuong", "dung"]

K = 0.5
V = len(LABELS)


# ==========================================
# 2. DOC CORPUS
# ==========================================

def load_corpus(path):
    sentences = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()

            # Bo qua dong trong va dong ghi chu
            if not line or line.startswith("#"):
                continue

            words = line.split()

            # Kiem tra tu co nam trong tu vung khong
            for word in words:
                if word not in LABELS:
                    raise ValueError(
                        f"Tu '{word}' khong nam trong tu vung."
                    )

            sentences.append(words)

    return sentences


# ==========================================
# 3. DEM UNIGRAM VA BIGRAM
# ==========================================

def count_ngrams(sentences):
    unigram_counts = Counter()
    bigram_counts = Counter()
    context_counts = Counter()

    for words in sentences:

        # Dem so lan xuat hien cua tung tu
        unigram_counts.update(words)

        # Dem cac cap tu lien tiep trong tung cau
        for i in range(len(words) - 1):

            previous_word = words[i]
            current_word = words[i + 1]

            bigram_counts[
                (previous_word, current_word)
            ] += 1

            # Tong so chuyen tiep tu previous_word
            context_counts[previous_word] += 1

    return unigram_counts, bigram_counts, context_counts


# ==========================================
# 4. XAC SUAT BIGRAM ADD-K
# ==========================================

def bigram_probability(
    previous_word,
    current_word,
    bigram_counts,
    context_counts,
    k=K
):
    numerator = (
        bigram_counts[(previous_word, current_word)] + k
    )

    denominator = (
        context_counts[previous_word] + k * V
    )

    return numerator / denominator


# ==========================================
# 5. CHAY CHUONG TRINH
# ==========================================

def main():

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Khong tim thay corpus_bigram.txt"
        )

    sentences = load_corpus(CORPUS_PATH)

    unigram_counts, bigram_counts, context_counts = (
        count_ngrams(sentences)
    )

    print("=== CORPUS ===")
    for sentence in sentences:
        print(" ".join(sentence))

    print("\n=== UNIGRAM COUNTS ===")
    for word in LABELS:
        print(f"{word:6s}: {unigram_counts[word]}")

    print("\n=== BIGRAM COUNTS ===")
    for (previous, current), count in sorted(
        bigram_counts.items()
    ):
        print(f"({previous}, {current}): {count}")

    print("\n=== BIGRAM PROBABILITIES: ADD-K ===")
    print("k =", K)
    print("V =", V)

    # Minh hoa cac tu tiep theo cua "mo"
    previous_word = "mo"

    print(
        f"\nP(word tiep theo | {previous_word}):"
    )

    for current_word in LABELS:

        probability = bigram_probability(
            previous_word,
            current_word,
            bigram_counts,
            context_counts
        )

        print(
            f"P({current_word} | {previous_word}) "
            f"= {probability:.4f}"
        )

    # Kiem tra tong xac suat
    total_probability = sum(
        bigram_probability(
            previous_word,
            current_word,
            bigram_counts,
            context_counts
        )
        for current_word in LABELS
    )

    print(
        "\nTong xac suat:",
        round(total_probability, 6)
    )

    print("\nDA HOAN THANH BUOC 9.1!")


if __name__ == "__main__":
    main()
