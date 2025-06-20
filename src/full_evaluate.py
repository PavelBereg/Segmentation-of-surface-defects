# src/full_evaluate.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from glob import glob
from albumentations import Resize
import argparse

import config
from model_setup import val_transform


def visualize_single_prediction(model, image_path, mask_path, category, device):
    """
    Функция для предсказания и визуализации одного изображения.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if os.path.exists(mask_path):
        gt_mask_binary = cv2.imread(mask_path, 0)
        gt_mask_binary = (gt_mask_binary > 0).astype(np.float32)
    else:
        gt_mask_binary = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    image_size = config.IMG_SIZE

    transformed = val_transform(image=image, mask=gt_mask_binary)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pr_mask_tensor = model(image_tensor)
        pr_mask = torch.argmax(pr_mask_tensor, dim=1).squeeze().cpu().numpy()

    resizer = Resize(height=image_size, width=image_size)
    resized_orig = resizer(image=image)['image']
    resized_gt = resizer(image=gt_mask_binary)['image']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    defect_type = os.path.basename(os.path.dirname(image_path))
    fig.suptitle(f"Категория: {category} | Дефект: {defect_type} | Файл: {os.path.basename(image_path)}", fontsize=14)

    axes[0].imshow(resized_orig);
    axes[0].set_title('Исходное изображение');
    axes[0].axis('off')
    axes[1].imshow(resized_gt, cmap='gray');
    axes[1].set_title('Истинная маска');
    axes[1].axis('off')
    axes[2].imshow((pr_mask > 0), cmap='gray');
    axes[2].set_title('Предсказанная маска (бинарная)');
    axes[2].axis('off')

    return fig


def main(args):
    category = args.category
    data_dir = args.data_dir
    model_path = f'../best_model_{category}.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- РАСШИРЕННОЕ ТЕСТИРОВАНИЕ ДЛЯ КАТЕГОРИИ: '{category.upper()}' ---")

    print(f"\n[Шаг 1/3] Загрузка модели из {model_path}...")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print("  - Модель успешно загружена.")
    except FileNotFoundError:
        print(f"!!! ОШИБКА: Файл модели '{model_path}' не найден.")
        print(f"!!!         Убедитесь, что вы обучили модель для этой категории.")
        return

    print(f"\n[Шаг 2/3] Поиск типов дефектов...")
    test_path = os.path.join(data_dir, category, 'test')
    defect_types = sorted(
        [os.path.basename(f) for f in glob(os.path.join(test_path, '*')) if os.path.isdir(f) and 'good' not in f])

    if not defect_types:
        print(f"!!! В категории '{category}' не найдено папок с дефектами.")
        return

    print(f"  - Найденные типы дефектов: {defect_types}")

    print("\n[Шаг 3/3] Запуск пакетного режима...")
    results_dir = os.path.join('..', 'results', category)
    os.makedirs(results_dir, exist_ok=True)
    print(f"  - Результаты будут сохранены в: {os.path.abspath(results_dir)}")

    for defect_type in defect_types:
        defect_images = sorted(glob(os.path.join(test_path, defect_type, '*.png')))
        if not defect_images:
            continue

        image_path = defect_images[0]
        mask_name = os.path.basename(image_path).replace('.png', '_mask.png')
        mask_path = os.path.join(data_dir, category, 'ground_truth', defect_type, mask_name)

        print(f"  - Обработка дефекта '{defect_type}'...")
        fig = visualize_single_prediction(model, image_path, mask_path, category, device)

        save_path = os.path.join(results_dir, f"result_{defect_type}.png")
        fig.savefig(save_path)
        plt.close(fig)

    print("\n--- ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ---")
    print(f"Все результаты сохранены в папке {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуальная оценка многоклассовой модели сегментации.")
    parser.add_argument('--category', type=str, required=True, help="Название категории.")
    parser.add_argument('--data_dir', type=str, default='../data', help="Путь к данным.")
    parsed_args = parser.parse_args()
    main(parsed_args)