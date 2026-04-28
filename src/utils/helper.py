import os
import random
import cv2

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



def convert_to_yolo(input_file, image_dir, output_dir, class_id=0):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as f:
        lines = f.readlines()

    lines = lines[2:]

    processed_images = set()

    for line in lines:
        parts = line.strip().split()
        image_name = parts[0]
        x_min = float(parts[1])
        y_min = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không đọc được ảnh: {image_name}")
            continue

        img_h, img_w = img.shape[:2]

        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w /= img_w
        h /= img_h

        label_path = os.path.join(output_dir, image_name.replace(".jpg", ".txt"))

        if image_name not in processed_images:
            if os.path.exists(label_path):
                os.remove(label_path)
            processed_images.add(image_name)

        # append bbox
        with open(label_path, "a") as out:
            out.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

    print("done")


def visualize_and_save(image_path, label_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh")
        return

    h, w = img.shape[:2]

    img_draw = img.copy()

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x_min = int(x_center - bw / 2)
        y_min = int(y_center - bh / 2)
        x_max = int(x_center + bw / 2)
        y_max = int(y_center + bh / 2)

        cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    file_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, file_name)

    cv2.imwrite(save_path, img_draw)

    print(f"Đã lưu: {save_path}")


def visualize_folder(image_dir, label_dir, output_dir, limit=30):
    images = os.listdir(image_dir)[:limit]

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        visualize_and_save(img_path, label_path, output_dir)
def visualize_bbox_from_txt(txt_file, image_dir, output_dir, limit=10):
    os.makedirs(output_dir, exist_ok=True)

    with open(txt_file, "r") as f:
        lines = f.readlines()

    # bỏ 2 dòng đầu (CelebA)
    lines = lines[2:]

    count = 0

    for line in lines:
        if count >= limit:
            break

        parts = line.strip().split()
        image_name = parts[0]
        x_min = int(parts[1])
        y_min = int(parts[2])
        w = int(parts[3])
        h = int(parts[4])

        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print("Không đọc được:", image_name)
            continue

        # copy ảnh
        img_draw = img.copy()

        # tính góc phải dưới
        x_max = x_min + w
        y_max = y_min + h

        # vẽ bbox
        cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # lưu
        save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(save_path, img_draw)

        print("Saved:", save_path)
        count += 1


if __name__ == "__main__":
    visualize_bbox_from_txt(
        r"E:\OU\Learning\AI\btl\FaID-System\data\detection\raw\list_bbox_celeba.txt",
        r"E:\OU\Learning\AI\btl\FaID-System\data\detection\raw\celeb_a",
        r"E:\OU\Learning\AI\btl\FaID-System\data\detection\res",
        limit=10
    )

    # visualize_folder(r"E:\OU\Learning\AI\btl\FaID-System\data\detection\raw\celeb_a",
    #                  r"E:\OU\Learning\AI\btl\FaID-System\data\detection\convert",
    #                  r"E:\OU\Learning\AI\btl\FaID-System\data\detection\res")
    # # visualize_and_save(
    #     r"E:\OU\Learning\AI\btl\FaID-System\data\detection\raw\celeb_a\000002.jpg",
    #     r"E:\OU\Learning\AI\btl\FaID-System\data\detection\convert\000002.txt",
    #     r"E:\OU\Learning\AI\btl\FaID-System\data\detection\res"
    # )

    # convert_to_yolo(
    #     "E:/OU/Learning/AI/btl/FaID-System/data/detection/raw/list_bbox_celeba.txt",
    #     "E:/OU/Learning/AI/btl/FaID-System/data/detection/raw/celeb_a",
    #     "E:/OU/Learning/AI/btl/FaID-System/data/detection/convert"
    # )