# src/model_setup.py (ВЕРСИЯ ДЛЯ НОВОГО API)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import config

# Аугментации остаются сильными
train_transform = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=0.5, p=1)
    ], p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def get_model():
    """Собирает модель U-Net с Dropout в декодере."""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        # --- ДОБАВЛЯЕМ DROPOUT И ВНИМАНИЕ ---
        decoder_attention_type='scse',
        decoder_dropout=0.3
    ).to(config.DEVICE)
    return model