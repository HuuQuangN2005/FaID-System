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

    with open(LANDMARKS_FILE, "r") as f:
        _ = f.readline()
        header_columns = f.readline()
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    random.seed(42)
    random.shuffle(lines)

    total = len(lines)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]

    def save_file(name, data, columns):
        path = os.path.join(SPLITS_FOLDER, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{len(data)}\n")
            f.write(columns if columns.endswith('\n') else columns + '\n')
            f.write("\n".join(data))

    save_file("train.txt", train_data, header_columns)
    save_file("val.txt", val_data, header_columns)
    save_file("test.txt", test_data, header_columns)

if __name__ == "__main__":
    split_data()