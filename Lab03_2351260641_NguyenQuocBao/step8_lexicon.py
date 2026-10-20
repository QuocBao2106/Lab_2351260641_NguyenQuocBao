
from pathlib import Path

LEXICON_PATH = Path("lexicon.txt")

EXPECTED_WORDS = [
    "mo",
    "dong",
    "len",
    "xuong",
    "dung"
]


def parse_lexicon(path):
    """
    Doc lexicon.txt va chuyen thanh dictionary.

    Moi dong co dang:
    WORD PHONE_1 PHONE_2 ...
    """

    lexicon = {}

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Bo qua dong trong va dong ghi chu
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Moi dong phai co tu va it nhat mot phone
            if len(parts) < 2:
                raise ValueError(
                    f"Dong {line_number} khong hop le: {line}"
                )

            word = parts[0]
            phones = parts[1:]

            # Kiem tra tu bi lap
            if word in lexicon:
                raise ValueError(
                    f"Tu '{word}' bi lap trong lexicon.txt"
                )

            lexicon[word] = phones

    return lexicon


def main():

    if not LEXICON_PATH.exists():
        raise FileNotFoundError(
            "Khong tim thay file lexicon.txt"
        )

    lexicon = parse_lexicon(LEXICON_PATH)

    print("=== LEXICON DA DOC ===")

    for word, phones in lexicon.items():
        print(f"{word} -> {phones}")

    # Kiem tra da co du 5 tu chua
    missing_words = [
        word for word in EXPECTED_WORDS
        if word not in lexicon
    ]

    if missing_words:
        print("\nCac tu con thieu:", missing_words)
    else:
        print("\nDa co du 5 tu trong lexicon.")

    # Canh bao neu van dang dung phone minh hoa
    using_placeholders = any(
        phone.startswith("PH_")
        for phones in lexicon.values()
        for phone in phones
    )

    if using_placeholders:
        print(
            "\nCANH BAO: Lexicon dang dung phone minh hoa."
        )
        print(
            "Can thay bang phone chinh thuc "
            "truoc khi nop bai."
        )

    print("\nDictionary Python:")
    print(lexicon)


if __name__ == "__main__":
    main()
