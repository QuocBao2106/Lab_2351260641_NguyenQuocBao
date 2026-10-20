
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3

# 1. Liệt kê các thiết bị có thể thu âm
devices = sd.query_devices()
input_devices = []

print("=== DANH SACH MICROPHONE ===")

for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        input_devices.append(i)
        print(f"ID: {i} | Ten: {dev['name']}")

if not input_devices:
    raise RuntimeError(
        "Khong tim thay microphone. "
        "Hay kiem tra ket noi va quyen truy cap micro."
    )

# 2. Chọn microphone
default_input = sd.default.device[0]
print("\nMicro mac dinh:", default_input)

choice = input(
    "Nhap ID microphone (Enter de dung mac dinh): "
).strip()

if choice:
    device = int(choice)

    if device not in input_devices:
        raise ValueError("ID da chon khong phai microphone hop le.")
else:
    if default_input in input_devices:
        device = default_input
    else:
        device = input_devices[0]

print("Dang su dung:", sd.query_devices(device)["name"])

# 3. Thu âm thử trong 3 giây
input("\nNhan Enter de bat dau thu am...")

print("DANG THU AM! Hay noi thu mot cau trong 3 giay.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=device
)

sd.wait()

# 4. Lưu file WAV mono, PCM 16-bit
sf.write(
    "mic_test.wav",
    audio,
    SAMPLE_RATE,
    subtype="PCM_16"
)

# 5. Kiểm tra tín hiệu thu được
peak = np.max(np.abs(audio.astype(np.int32)))

print("\nDa luu file mic_test.wav")
print("Tan so lay mau:", SAMPLE_RATE, "Hz")
print("So kenh:", audio.shape[1])
print("So mau thu duoc:", len(audio))
print("Bien do lon nhat:", peak)

if peak < 300:
    print("Can kiem tra lai micro vi tin hieu thu duoc rat nho.")
else:
    print("Da thu duoc tin hieu. Hay nghe lai file de xac nhan.")
