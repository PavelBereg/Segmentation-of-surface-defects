# src/config.py (ИЗМЕНЕННАЯ ВЕРСИЯ)

import torch

# Настройки путей
DATA_DIR = '../data/'
#CATEGORY = 'carpet'
#MODEL_PATH = f'../best_model_{CATEGORY}.pth'

# Параметры модели и обучения
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001