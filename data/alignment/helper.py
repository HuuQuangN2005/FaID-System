import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(BASE_DIR, "raw")
SPLITS_FOLDER = os.path.join(BASE_DIR, "splits")
LANDMARKS_FILE = os.path.join(RAW_FOLDER, "list_landmarks_align_celeba.txt")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

def split_data():
    os.makedirs(SPLITS_FOLDER, exist_ok=True)

    with open(LANDMARKS_FILE, "r", encoding="utf-8") as f:
        total_count = f.readline().strip()
        header_columns = f.readline().strip()
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")

    random.seed(42)
    random.shuffle(lines)

    total = len(lines)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]

    print(f"Train: {len(train_data)} ({len(train_data)/total*100:.2f}%)")
    print(f"Val:   {len(val_data)}   ({len(val_data)/total*100:.2f}%)")
    print(f"Test:  {len(test_data)}  ({len(test_data)/total*100:.2f}%)")

    def save_split(name, data):
        path = os.path.join(SPLITS_FOLDER, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(header_columns + "\n")
            if data:
                f.write("\n".join(data) + "\n")
        print(f"Saved: {path} ({len(data)} lines)")

    save_split("train.txt", train_data)
    save_split("val.txt", val_data)
    save_split("test.txt", test_data)

if __name__ == "__main__":
    split_data()