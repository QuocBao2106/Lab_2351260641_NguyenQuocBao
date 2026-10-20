
from pathlib import Path

# 5 nhãn lệnh tiếng nói
labels = ["mo", "dong", "len", "xuong", "dung"]

# Tạo các thư mục train và test
for split in ["train", "test"]:
    for label in labels:
        folder = Path("dataset") / split / label
        folder.mkdir(parents=True, exist_ok=True)

print("Da tao xong cau truc du lieu!\n")

# Hiển thị các thư mục đã tạo
for folder in sorted(Path("dataset").rglob("*")):
    if folder.is_dir():
        print(folder)
