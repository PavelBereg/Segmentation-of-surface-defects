# Система семантической сегментации промышленных дефектов

Проект представляет собой реализацию системы для автоматического контроля качества в промышленности. Модель на основе глубокого обучения способна находить и попиксельно выделять дефекты на изображениях различных материалов.

Работа выполнена в рамках выпускной квалификационной работы.


*Пример работы модели на категории 'carpet'. Замените эту ссылку на скриншот с вашим результатом.*

## Ключевые особенности

- **Архитектура:** U-Net с предобученными бэкбонами (например, `EfficientNet-B0` или `ResNet50`) из библиотеки `segmentation-models-pytorch`.
- **Data-Centric подход:** Для решения проблемы отсутствия дефектов в обучающей выборке датасета MVTec AD, применена стратегия формирования обучающей выборки из части тестовых данных.
- **Борьба с переобучением:** Реализован комплексный подход, включающий:
  - Интенсивную аугментацию данных с помощью `Albumentations`.
  - Двухэтапную fine-tuning стратегию (сначала "голова", затем вся модель).
  - L2-регуляризацию (Weight Decay) и планировщик скорости обучения.
- **Гибкость:** Скрипты для обучения и тестирования являются универсальными и могут работать с любой из 15 категорий датасета.


## Быстрый старт

### 1. Клонирование и установка

git clone https://github.com/PavelBereg/Segmentation-of-surface-defects.git
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
2. Подготовка данных

Скачайте датасет MVTec Anomaly Detection.

Распакуйте архив так, чтобы папки с категориями (bottle, carpet и т.д.) находились внутри директории data/.

3. Обучение модели

Запустите интерактивный скрипт из корневой папки проекта:

python src/train.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Далее в консоли появится меню для выбора категории (или нескольких) для обучения. Обученная модель будет сохранена в корневую директорию проекта (например, best_model_carpet.pth).

4. Тестирование и оценка

После обучения модели для нужной категории, запустите скрипты для оценки.

Визуальная оценка:

# Пример для категории 'carpet'
python src/full_evaluate.py --category carpet
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Изображения с результатами будут сохранены в папку results/carpet/.

Количественная оценка (расчет метрик):

# Пример для категории 'carpet'
python src/calculate_metrics.py --category carpet
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Результаты (средний IoU) будут выведены в консоль.
