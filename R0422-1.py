import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 基本設定
# =========================
random.seed(42)
torch.manual_seed(42)

DEVICE = "cpu"
print("device =", DEVICE)
print("torch version =", torch.__version__)

# 原始資料夾
OK_SRC = Path(r"C:\Users\cryst\Downloads\2OK.coco-segmentation\train")
FAIL_SRC = Path(r"C:\Users\cryst\Downloads\2fail.coco-segmentation\train")

# 自動切分後的資料夾
BASE_SPLIT_DIR = Path(r"C:\code\picture_labelROI_cls0424")

TRAIN_DIR = BASE_SPLIT_DIR / "train"
VALID_DIR = BASE_SPLIT_DIR / "valid"
TEST_DIR  = BASE_SPLIT_DIR / "test"

# 輸出資料夾
OUTPUT_DIR = Path(r"C:\code\resnet_ok_fail_output0506")
MIS_DIR = OUTPUT_DIR / "misclassified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIS_DIR.mkdir(parents=True, exist_ok=True)

# 圖片副檔名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 切分比例
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO  = 0.15

# 訓練參數
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4


# =========================
# 工具函式
# =========================
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def collect_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    return files


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_and_make_split_dirs(base_dir: Path):
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ["train", "valid", "test"]:
        for cls in ["OK", "fail"]:
            safe_mkdir(base_dir / split / cls)


def split_list(items, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-8
    items = items[:]
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_items = items[:n_train]
    valid_items = items[n_train:n_train + n_valid]
    test_items = items[n_train + n_valid:]

    return train_items, valid_items, test_items


def copy_files(files, dst_dir: Path):
    safe_mkdir(dst_dir)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def prepare_dataset():
    ok_files = collect_images(OK_SRC)
    fail_files = collect_images(FAIL_SRC)

    print(f"OK 圖片數量   = {len(ok_files)}")
    print(f"fail 圖片數量 = {len(fail_files)}")

    clear_and_make_split_dirs(BASE_SPLIT_DIR)

    ok_train, ok_valid, ok_test = split_list(ok_files, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    fail_train, fail_valid, fail_test = split_list(fail_files, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)

    copy_files(ok_train, TRAIN_DIR / "OK")
    copy_files(ok_valid, VALID_DIR / "OK")
    copy_files(ok_test,  TEST_DIR  / "OK")

    copy_files(fail_train, TRAIN_DIR / "fail")
    copy_files(fail_valid, VALID_DIR / "fail")
    copy_files(fail_test,  TEST_DIR  / "fail")

    print("資料切分完成")
    print(f"train/OK   = {len(ok_train)}")
    print(f"valid/OK   = {len(ok_valid)}")
    print(f"test/OK    = {len(ok_test)}")
    print(f"train/fail = {len(fail_train)}")
    print(f"valid/fail = {len(fail_valid)}")
    print(f"test/fail  = {len(fail_test)}")


def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(VALID_DIR, transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("class_to_idx =", train_ds.class_to_idx)

    return train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


def train_model(model, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_path = OUTPUT_DIR / "best_resnet18.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        y_true = []
        y_pred = []

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(y_true, y_pred)

        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(">>> 已儲存最佳模型")

    print("最佳驗證準確率 =", best_val_acc)
    return best_path


def save_misclassified_images(model, test_ds):
    idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}

    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    mis_count = 0

    for path, true_label in test_ds.samples:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"無法開啟圖片: {path}, error={e}")
            continue

        x = tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)
            pred_label = out.argmax(dim=1).item()

        if pred_label != true_label:
            true_name = idx_to_class[true_label]
            pred_name = idx_to_class[pred_label]

            dst_dir = MIS_DIR / f"true_{true_name}_pred_{pred_name}"
            safe_mkdir(dst_dir)

            stem = Path(path).stem
            suffix = Path(path).suffix
            dst_path = dst_dir / f"{stem}{suffix}"

            shutil.copy2(path, dst_path)
            mis_count += 1

    print("判錯圖片數量 =", mis_count)


def test_model(model, test_loader, test_ds):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    precision = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}

    print("\n===== Test Result =====")
    print(f"Test Loss  : {test_loss:.4f}")
    print(f"Accuracy   : {test_acc:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("class mapping:", idx_to_class)

    result_txt = OUTPUT_DIR / "test_metrics.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("===== Test Result =====\n")
        f.write(f"Test Loss  : {test_loss:.4f}\n")
        f.write(f"Accuracy   : {test_acc:.4f}\n")
        f.write(f"Precision  : {precision:.4f}\n")
        f.write(f"Recall     : {recall:.4f}\n")
        f.write(f"F1-score   : {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f"class mapping: {idx_to_class}\n")

    save_misclassified_images(model, test_ds)


def main():
    prepare_dataset()

    train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader = get_dataloaders()

    model = build_model(num_classes=2)

    best_model_path = train_model(model, train_loader, valid_loader)

    print(f"\n載入最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_model(model, test_loader, test_ds)

    print("\n全部完成")
    print("切分資料夾：", BASE_SPLIT_DIR)
    print("模型與結果：", OUTPUT_DIR)


if __name__ == "__main__":
    main()