import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from glob import glob

import config
from dataset import MVTecDataset
from model_setup import get_model, train_transform, val_transform


# --- Функции ---
def multiclass_iou_score(y_pred, y_true, num_classes):
    """
    Вычисляет средний IoU по всем классам дефектов (пропуская фон).
    """
    y_pred = torch.argmax(y_pred, dim=1)

    iou_per_class = []
    for i in range(1, num_classes):
        pred_i = (y_pred == i)
        true_i = (y_true == i)

        intersection = torch.logical_and(pred_i, true_i).sum()
        union = torch.logical_or(pred_i, true_i).sum()

        if union == 0:
            iou_i = 1.0
        else:
            iou_i = (intersection.float() / union.float()).item()
        iou_per_class.append(iou_i)

    return np.mean(iou_per_class) if iou_per_class else 1.0


class MulticlassCombinedLoss(torch.nn.Module):
    """
    Комбинированная функция потерь BCE + Dice для многоклассовой сегментации.
    """

    def __init__(self):
        super(MulticlassCombinedLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, y_pred, y_true):
        ce = self.ce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        return 0.5 * ce + 0.5 * dice


# --- Главная функция обучения для ОДНОЙ категории ---
def train_one_category(category):
    print(f"\n{'=' * 20} НАЧАЛО ОБУЧЕНИЯ ДЛЯ КАТЕГОРИИ: {category.upper()} {'=' * 20}")

    print("\n[Шаг 1/5] Создание датасетов...")
    try:
        train_dataset = MVTecDataset(config.DATA_DIR, category, is_train=True, transform=train_transform)
        val_dataset = MVTecDataset(config.DATA_DIR, category, is_train=False, transform=val_transform)

        num_classes = train_dataset.num_classes
        if num_classes <= 1:
            print(f"!!! В категории '{category}' не найдено дефектов для обучения. Пропускаем.")
            return

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
                                pin_memory=True)
        print("  - Загрузчики данных успешно созданы.")
    except FileNotFoundError as e:
        print(f"!!! ОШИБКА: {e}. Пропускаем категорию '{category}'.")
        return

    print("\n[Шаг 2/5] Создание модели...")
    model = get_model(num_classes=num_classes)
    loss_fn = MulticlassCombinedLoss()
    print(f"  - Модель создана для {num_classes} классов.")

    best_iou_score = 0.0
    model_save_path = f'../best_model_{category}.pth'

    print("\n[Шаг 3/5] ЭТАП 1: Обучение только декодера...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.LEARNING_RATE)

    num_head_epochs = 5
    for i in range(num_head_epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Обучение головы {i + 1}/{num_head_epochs}"):
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

    print("\n[Шаг 4/5] ЭТАП 2: Обучение всей модели (fine-tuning)...")
    for param in model.encoder.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 100, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    for i in range(config.EPOCHS):
        model.train()
        train_loss, train_iou_list = 0.0, []
        for images, masks in tqdm(train_loader, desc=f"Обучение (FT) {i + 1}/{config.EPOCHS}"):
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou_list.append(multiclass_iou_score(outputs.detach(), masks, num_classes))

        avg_train_loss = np.mean(train_loss) / len(train_loader)
        avg_train_iou = np.mean(train_iou_list) if train_iou_list else 0.0

        model.eval()
        val_loss, val_iou_list = 0.0, []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Валидация (FT) {i + 1}/{config.EPOCHS}"):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE).long()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                if torch.sum(masks) > 0:
                    val_iou_list.append(multiclass_iou_score(outputs, masks, num_classes))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = np.mean(val_iou_list) if val_iou_list else 1.0

        scheduler.step()

        print(
            f"Эпоха {i + 1}: Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f} | Valid Loss: {avg_val_loss:.4f}, Valid IoU (defects only): {avg_val_iou:.4f}")

        if best_iou_score < avg_val_iou:
            best_iou_score = avg_val_iou
            torch.save(model, model_save_path)
            print(f"** Новое лучшее значение IoU: {best_iou_score:.4f}. Модель для '{category}' сохранена! **")

    print(f"\n[Шаг 5/5] Обучение для категории '{category}' завершено.")
    print(f"Лучший результат IoU: {best_iou_score:.4f}")


# --- ИНТЕРАКТИВНЫЙ ГЛАВНЫЙ БЛОК ЗАПУСКА ---
if __name__ == "__main__":
    all_categories = sorted([os.path.basename(p) for p in glob(os.path.join(config.DATA_DIR, '*')) if os.path.isdir(p)])

    if not all_categories:
        print(f"!!! ОШИБКА: Не найдено ни одной папки с категориями в директории '{config.DATA_DIR}'")
    else:
        while True:
            print("\nДоступные для обучения категории:")
            for i, category in enumerate(all_categories):
                print(f"  {i + 1}. {category}")
            print("  all. Обучить все категории")
            print("  exit. Выход")

            choice = input(
                "\nВведите номер категории, несколько номеров через запятую (например, 1,3), 'all' или 'exit': ").strip().lower()

            if choice == 'exit':
                print("Выход из программы.")
                break

            categories_to_train = []
            if choice == 'all':
                categories_to_train = all_categories
            else:
                try:
                    indices = [int(i.strip()) - 1 for i in choice.split(',')]
                    for idx in indices:
                        if 0 <= idx < len(all_categories):
                            categories_to_train.append(all_categories[idx])
                        else:
                            print(f"!!! Некорректный номер: {idx + 1}")
                    categories_to_train = sorted(list(set(categories_to_train)))
                except ValueError:
                    print("!!! Некорректный ввод. Пожалуйста, введите номера, 'all' или 'exit'.")
                    continue

            if categories_to_train:
                print(f"\nБудет запущено обучение для: {categories_to_train}")
                for category_name in categories_to_train:
                    train_one_category(category_name)
                print(f"\n{'=' * 20} ВСЕ ВЫБРАННЫЕ ЗАДАЧИ ВЫПОЛНЕНЫ {'=' * 20}")
            else:
                print("Не выбрано ни одной категории для обучения.")