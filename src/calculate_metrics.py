# src/calculate_metrics.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import config  # Нужен для BATCH_SIZE
from dataset import MVTecDataset
from model_setup import val_transform


# Наша собственная функция для IoU
def iou_score(y_pred, y_true, smooth=1e-7):
    y_pred = torch.sigmoid(y_pred).round()
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def calculate_metrics(args):
    category = args.category
    data_dir = args.data_dir
    model_path = f'../best_model_{category}.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- РАСЧЕТ МЕТРИК ДЛЯ КАТЕГОРИИ: '{category.upper()}' ---")

    print(f"\n[Шаг 1/3] Загрузка модели из {model_path}...")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print("  - Модель успешно загружена.")
    except FileNotFoundError:
        print(f"!!! ОШИБКА: Файл модели '{model_path}' не найден.")
        return

    print(f"\n[Шаг 2/3] Подготовка валидационного/тестового датасета...")
    # is_train=False, чтобы получить выборку из хороших и 20% дефектных примеров
    test_dataset = MVTecDataset(data_dir, category, is_train=False, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    print("\n[Шаг 3/3] Выполнение предсказаний и расчет метрик...")

    total_iou = 0.0
    with torch.no_grad():
        for images, gt_masks in tqdm(test_loader, desc=f"Тестирование '{category}'"):
            images, gt_masks = images.to(device), gt_masks.to(device)
            pr_masks = model(images)
            total_iou += iou_score(pr_masks, gt_masks).item()

    avg_iou = total_iou / len(test_loader)

    print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ ---")
    print(f"Категория: {category}")
    print("-------------------------------------------------")
    print(f"Средний IoU (mIoU): {avg_iou:.4f}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расчет метрик для обученной модели сегментации.")
    parser.add_argument('--category', type=str, required=True, help="Название категории для тестирования.")
    parser.add_argument('--data_dir', type=str, default='../data', help="Путь к корневой папке с данными.")
    parsed_args = parser.parse_args()
    calculate_metrics(parsed_args)