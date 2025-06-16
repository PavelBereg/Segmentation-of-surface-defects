# src/dataset.py (МНОГОКЛАССОВАЯ ВЕРСИЯ)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from sklearn.model_selection import train_test_split


class MVTecDataset(Dataset):
    def __init__(self, data_dir, category, is_train=True, split_ratio=0.8, transform=None):
        test_path = os.path.join(data_dir, category, 'test')
        good_files = sorted(glob(os.path.join(test_path, 'good', '*.png')))

        all_defect_folders = sorted(
            [f for f in glob(os.path.join(test_path, '*')) if os.path.isdir(f) and 'good' not in f])

        # --- Создаем словарь классов: фон - 0, дефекты - 1, 2, 3... ---
        self.defect_types = [os.path.basename(f) for f in all_defect_folders]
        self.class_map = {name: i + 1 for i, name in enumerate(self.defect_types)}
        print(f"  - Для категории '{category}' найдены классы: {self.class_map}")

        all_defect_files = []
        for folder in all_defect_folders:
            all_defect_files.extend(sorted(glob(os.path.join(folder, '*.png'))))

        if all_defect_files:
            train_defect_files, val_defect_files = train_test_split(all_defect_files, test_size=(1.0 - split_ratio),
                                                                    random_state=42)
        else:
            train_defect_files, val_defect_files = [], []

        if is_train:
            self.image_files = train_defect_files
            data_type = "TRAIN (defect only)"
        else:
            self.image_files = val_defect_files + good_files
            data_type = "VALIDATION (good + defect)"

        if not self.image_files:
            raise FileNotFoundError(f"Не удалось найти файлы для {data_type} выборки.")

        print(f"  - ({data_type}) Найдено изображений: {len(self.image_files)}")

        self.mask_dir = os.path.join(data_dir, category, 'ground_truth')
        self.transform = transform
        self.num_classes = len(self.defect_types) + 1  # +1 для фона

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Создаем целочисленную маску с индексами классов ---
        if 'good' in image_path:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)
        else:
            defect_type = os.path.basename(os.path.dirname(image_path))
            class_idx = self.class_map.get(defect_type, 0)

            mask_name = os.path.basename(image_path).replace('.png', '_mask.png')
            mask_path = os.path.join(self.mask_dir, defect_type, mask_name)

            if os.path.exists(mask_path):
                single_mask = cv2.imread(mask_path, 0)
                mask = (single_mask > 0).astype(np.int64) * class_idx
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Преобразуем в тензоры нужного типа
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()  # Маска с индексами должна быть Long

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

        return image, mask