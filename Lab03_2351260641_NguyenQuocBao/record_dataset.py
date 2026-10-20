
from pathlib import Path
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# ==============================
# 1. CAU HINH THU AM
# ==============================

SAMPLE_RATE = 16000       # Tan so lay mau: 16 kHz
RECORD_SECONDS = 2        # Moi lan thu am 2 giay
SILENCE_SECONDS = 0.3     # Them 0.3 giay im lang dau/cuoi

# Ten thu muc va cau lenh hien thi
LABELS = {
    "mo": "MO",
    "dong": "DONG",
    "len": "LEN",
    "xuong": "XUONG",
    "dung": "DUNG"
}

DATASET_DIR = Path("dataset")


# ==============================
# 2. CHON MICROPHONE
# ==============================

def choose_microphone():
    devices = sd.query_devices()

    input_devices = [
        i for i, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
    ]

    if not input_devices:
        raise RuntimeError("Khong tim thay microphone!")

    print("=== DANH SACH MICROPHONE ===")

    for i in input_devices:
        print(f"ID {i}: {devices[i]['name']}")

    default_input = sd.default.device[0]

    if default_input not in input_devices:
        default_input = input_devices[0]

    choice = input(
        f"\nNhap ID micro (Enter de dung ID {default_input}): "
    ).strip()

    if choice:
        device = int(choice)

        if device not in input_devices:
            raise ValueError("ID microphone khong hop le!")
    else:
        device = default_input

    print("Microphone dang su dung:",
          devices[device]["name"])

    return device


# ==============================
# 3. THU AM MOT LAN
# ==============================

def record_one_file(filepath, device):
    # Tao thu muc neu chua ton tai
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Neu file da ton tai, hoi truoc khi ghi de
    # tranh mat du lieu cu
    if filepath.exists():
        answer = input(
            f"File {filepath} da ton tai. "
            "Ghi de? (y/n): "
        ).strip().lower()

        if answer != "y":
            print("Giu nguyen file cu.")
            return

    input("Nhan Enter khi ban da san sang...")

    print("Chuan bi noi lenh:")

    for number in [3, 2, 1]:
        print(number)
        time.sleep(0.5)

    print("BAT DAU! Hay noi lenh mot lan.")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device
    )

    sd.wait()

    # Kiem tra am luong tin hieu
    peak = np.max(np.abs(audio.astype(np.int32)))

    # Them khoang im lang vao dau va cuoi
    silence_length = int(SILENCE_SECONDS * SAMPLE_RATE)

    silence = np.zeros(
        (silence_length, 1),
        dtype=np.int16
    )

    audio = np.concatenate(
        [silence, audio, silence],
        axis=0
    )

    # Luu WAV mono, PCM 16-bit
    sf.write(
        filepath,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print("Da luu:", filepath)
    print("Muc tin hieu lon nhat:", peak)

    if peak < 300:
        print(
            "CANH BAO: Tin hieu rat nho. "
            "Hay kiem tra lai file ghi am."
        )


# ==============================
# 4. THU AM 40 FILE
# ==============================

def main():
    print("=== CHUONG TRINH THU AM LAB 3 ===")
    print("5 lenh, moi lenh 8 file.")
    print("6 file train va 2 file test cho moi lenh.")
    print("Moi lan chi noi lenh hien tren man hinh.")
    print("Khong noi ca cau, khong doc lien tuc.")

    device = choose_microphone()

    input(
        "\nKiem tra micro xong, nhan Enter "
        "de bat dau thu am."
    )

    for label, command in LABELS.items():
        print("\n" + "=" * 40)
        print(f"BAT DAU THU LENH: {command}")
        print("=" * 40)

        for i in range(1, 9):

            # 6 file dau dung de huan luyen
            if i <= 6:
                split = "train"
            else:
                split = "test"

            filepath = (
                DATASET_DIR
                / split
                / label
                / f"{label}_{i:02d}.wav"
            )

            print(f"\nLenh: {command}")
            print(f"Lượt thu: {i}/8")
            print(f"Nhom: {split}")
            print(f"File: {filepath}")

            record_one_file(filepath, device)

    print("\n=== DA THU AM XONG ===")

    # Kiem tra so luong file cua tung nhan
    all_ok = True

    for label in LABELS:
        train_dir = DATASET_DIR / "train" / label
        test_dir = DATASET_DIR / "test" / label

        train_files = list(train_dir.glob("*.wav"))
        test_files = list(test_dir.glob("*.wav"))

        print(
            f"{label}: train = {len(train_files)}, "
            f"test = {len(test_files)}"
        )

        if len(train_files) < 6 or len(test_files) < 2:
            all_ok = False

    if all_ok:
        print("\nDa du so luong file toi thieu!")
        print("Hay kiem tra lai cac file am thanh.")
    else:
        print("\nCon thieu file. Hay kiem tra cac thu muc.")


if __name__ == "__main__":
    main()
