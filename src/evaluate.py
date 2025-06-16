# src/evaluate.py

import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from albumentations import Resize

import config
from dataset import MVTecDataset
from model_setup import val_transform


def visualize_prediction():
    print("--- НАЧАЛО ПРОЦЕССА ВИЗУАЛЬНОЙ ОЦЕНКИ ---")

    print(f"\n[Шаг 1/3] Загрузка модели из {config.MODEL_PATH}...")
    try:
        model = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        model.eval()
        print("  - Модель успешно загружена.")
    except FileNotFoundError:
        print(f"!!! ОШИБКА: Файл модели не найден. Сначала запустите train.py.")
        return

    print(f"\n[Шаг 2/3] Подготовка тестового датасета для категории '{config.CATEGORY}'...")
    # ИСПРАВЛЕНИЕ: Убираем лишний config.DATA_DIR
    test_dataset = MVTecDataset(config.DATA_DIR, config.CATEGORY, is_train=False, transform=None)

    defective_indices = [i for i, path in enumerate(test_dataset.image_files) if 'good' not in path]
    if not defective_indices:
        print(f"В тестовой выборке не найдено дефектных изображений.")
        return

    idx = random.choice(defective_indices)
    image, gt_mask = test_dataset[idx]
    image_path = test_dataset.image_files[idx]
    print(f"  - Выбрано изображение: {os.path.basename(image_path)}")

    print("\n[Шаг 3/3] Выполнение предсказания и отображение результатов...")

    transformed = val_transform(image=image.permute(1, 2, 0).numpy().astype(np.uint8))
    image_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pr_mask_tensor = model(image_tensor)
        pr_mask = (torch.sigmoid(pr_mask_tensor).squeeze().cpu().numpy().round())

    print("  - Готово! Открывается окно с результатами...")

    resizer = Resize(height=config.IMG_SIZE, width=config.IMG_SIZE)
    resized_orig = resizer(image=image.permute(1, 2, 0).numpy().astype(np.uint8))['image']
    resized_gt = resizer(image=gt_mask.squeeze().numpy())['image']

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Результат для: {os.path.basename(image_path)}", fontsize=16)
    plt.subplot(1, 3, 1);
    plt.imshow(resized_orig);
    plt.title('Исходное изображение');
    plt.axis('off')
    plt.subplot(1, 3, 2);
    plt.imshow(resized_gt, cmap='gray');
    plt.title('Истинная маска');
    plt.axis('off')
    plt.subplot(1, 3, 3);
    plt.imshow(pr_mask, cmap='gray');
    plt.title('Предсказанная маска');
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    visualize_prediction()