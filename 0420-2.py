import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ========= 路徑設定 =========
DATASET_ROOT = Path(r"C:\code\pictures_1\screw.yolov8 (2)")
WORK_ROOT = Path(r"C:\code\pictures_1\work_training04202")
OUTPUT_DIR = Path(r"C:\code\pictures_1\output\training04202_roi")

# ========= 訓練參數 =========
MODEL_NAME = "yolov8s.pt"
EPOCHS = 30
IMGSZ = 1280
BATCH = 2
CONF_THRES = 0.1
IOU_THRES = 0.5

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}


def read_img_unicode(path):
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".heic":
            img = Image.open(str(path)).convert("RGB")
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[WARN] 圖片讀取失敗: {path} | {e}")
        return None


def write_img_unicode(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        path = path.with_suffix(".jpg")
        ext = ".jpg"

    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def find_dataset_root(root_dir):
    if not root_dir.exists():
        raise FileNotFoundError(f"找不到資料夾: {root_dir}")

    for p in root_dir.rglob("data.yaml"):
        return p.parent

    for p in root_dir.rglob("train"):
        if (p / "images").exists() and (p / "labels").exists():
            return p.parent

    raise FileNotFoundError("找不到 data.yaml 或 train/images + train/labels")


def count_images(img_dir):
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def list_images(img_dir):
    if not img_dir.exists():
        return []
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def print_dataset_status(dataset_dir):
    print("===== 資料集檢查 =====")
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_count = count_images(img_dir)
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"[CHECK] {split}: images={img_count}, labels={lbl_count}")
    print("=====================")


def choose_split_paths(dataset_dir):
    train_img_dir = dataset_dir / "train" / "images"
    valid_img_dir = dataset_dir / "valid" / "images"
    test_img_dir = dataset_dir / "test" / "images"

    train_count = count_images(train_img_dir)
    valid_count = count_images(valid_img_dir)
    test_count = count_images(test_img_dir)

    if train_count == 0:
        raise RuntimeError(f"train/images 沒有圖片: {train_img_dir}")

    val_path = valid_img_dir if valid_count > 0 else train_img_dir
    test_path = test_img_dir if test_count > 0 else None

    return train_img_dir, val_path, test_path


def rewrite_data_yaml(dataset_dir):
    yaml_path = dataset_dir / "data.yaml"
    train_path, val_path, test_path = choose_split_paths(dataset_dir)

    data = {
        "train": str(train_path.resolve()),
        "val": str(val_path.resolve()),
        "nc": 1,
        "names": ["screw"],
    }

    if test_path is not None:
        data["test"] = str(test_path.resolve())

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] 已重寫 data.yaml: {yaml_path}")
    print(f"[INFO] train = {data['train']}")
    print(f"[INFO] val   = {data['val']}")
    if "test" in data:
        print(f"[INFO] test  = {data['test']}")

    return yaml_path


def train_model(data_yaml_path):
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        workers=0,
        project=str(WORK_ROOT / "runs"),
        name="screw_det",
        exist_ok=True,
    )

    best_path = WORK_ROOT / "runs" / "screw_det" / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"訓練完成但找不到 best.pt: {best_path}")
    return best_path


def get_infer_images(dataset_dir):
    for split in ["test", "valid", "train"]:
        img_dir = dataset_dir / split / "images"
        imgs = list_images(img_dir)
        if imgs:
            print(f"[INFO] 使用 {split}/images 做推論輸出，共 {len(imgs)} 張")
            return imgs
    return []


def detect_and_crop(best_model_path, image_paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best_model_path))

    for img_path in image_paths:
        img = read_img_unicode(img_path)
        if img is None:
            print(f"[WARN] 讀取失敗: {img_path}")
            continue

        results = model.predict(
            source=img,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print(f"[INFO] 無偵測: {img_path.name}")
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        print(f"{img_path.name}")
        print(f"  螺絲數量: {len(boxes)}")

        for i, (box, conf) in enumerate(zip(boxes, confs), start=1):
            x1, y1, x2, y2 = box.astype(int)

            h, w = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            save_name = f"{img_path.stem}_roi_{i}_{conf:.2f}.jpg"
            save_path = output_dir / save_name
            write_img_unicode(save_path, roi)

        print("-" * 40)


def main():
    dataset_dir = find_dataset_root(DATASET_ROOT)
    print(f"[INFO] dataset_dir = {dataset_dir}")

    print_dataset_status(dataset_dir)

    data_yaml_path = rewrite_data_yaml(dataset_dir)

    best_model_path = train_model(data_yaml_path)
    print(f"[INFO] best model = {best_model_path}")

    infer_images = get_infer_images(dataset_dir)
    if not infer_images:
        raise RuntimeError("找不到可推論的圖片")

    detect_and_crop(best_model_path, infer_images, OUTPUT_DIR)
    print(f"[INFO] 完成，ROI輸出在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()