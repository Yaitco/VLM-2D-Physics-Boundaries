# Colab Runbook

## Что это
Короткая инструкция по запуску текущего упрощённого pipeline в Google Colab.

Рекомендуемый ноутбук:
[Unified_VLM_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/notebooks/Unified_VLM_Validation_Colab.ipynb)

Старые специализированные ноутбуки перенесены в:
[archive_review/notebooks_legacy](/home/alexander/Projects/VLM-2D-Physics-Boundaries/archive_review/notebooks_legacy)

## Перед запуском

### 1. Colab Secrets
Для текущего публичного репозитория GitHub-секрет больше не нужен.

Если хочешь логирование в Comet, можно добавить в Colab Secrets:
- `comet_api_key`
- `comet_workspace`
- `comet_project_name`

Без этих ключей ноутбук тоже запустится, просто Comet будет отключён.

### 2. Runtime
Рекомендуется GPU runtime.

Для моделей:
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`

лучше использовать Colab с заметным запасом VRAM.

## Основные настройки в ноутбуке

### Датасет
Для natural background subset:
```python
DATASET_NAME = "abo_physics_natural_bg_v2"
PROTOCOL_NAME = "natural_bg_v2"
```

Для ABO150:
```python
DATASET_NAME = "abo_150_expanded"
PROTOCOL_NAME = "narrow_core"
```

### Протокол
```python
PROTOCOL_NAME = "narrow_core"
```

Доступные варианты:
- `narrow_core`
- `full_expanded`
- `pdf_compact`
- `natural_bg_v2`

Рекомендуемый старт:
- `narrow_core`

### Модель
```python
SELECTED_MODEL = "qwen2_5_vl_7b"
```

### Визуальные варианты
```python
EVAL_VARIANTS = ["raw"]
```

Если у samples нет `mask_path`, реально доступен только `raw`.

### Запуск на вручную approved subset
Если после ручного review у тебя есть:
`dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_approved_meta.json`

можно запускать pipeline прямо на нём:

```python
META_OVERRIDE_PATH = "dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_approved_meta.json"
```

Тогда основной `meta.json` не трогается, а loader берёт только approved-версию.

### Zero-shot / few-shot
```python
FEW_SHOT_K = 0
FEW_SHOT_SELECTION_MODE = "fixed"
```

Рекомендуемые значения:
- `FEW_SHOT_K = 0` для zero-shot baseline
- `FEW_SHOT_K = 2` для первого few-shot сравнения
- `FEW_SHOT_SELECTION_MODE = "fixed"` для более быстрого few-shot

### Размер батча по свойствам
```python
PROPERTY_BATCH_SIZE = 8
```

Практически:
- `8` — безопасный старт
- `12` или `16` — хороший следующий шаг, если хватает памяти

### Размер подвыборки
```python
MAX_SAMPLES = 50
RANDOM_SEED = 42
```

Так удобно делать быстрые сравнения моделей на одном и том же subset.

## Рекомендуемый порядок прогонов

### Первый baseline
```python
PROTOCOL_NAME = "narrow_core"
SELECTED_MODEL = "qwen2_5_vl_7b"
EVAL_VARIANTS = ["raw"]
FEW_SHOT_K = 0
PROPERTY_BATCH_SIZE = 8
MAX_SAMPLES = 50
```

### Первый few-shot baseline
```python
PROTOCOL_NAME = "narrow_core"
SELECTED_MODEL = "qwen2_5_vl_7b"
EVAL_VARIANTS = ["raw"]
FEW_SHOT_K = 2
FEW_SHOT_SELECTION_MODE = "fixed"
PROPERTY_BATCH_SIZE = 8
MAX_SAMPLES = 50
```

### Сравнение нескольких моделей
```python
RUN_MULTI_MODEL = True
MULTI_MODEL_KEYS = ["qwen3_vl_8b", "qwen2_5_vl_7b", "llava_onevision_1_5_8b"]
```

### Полный протокол для лучших моделей
```python
PROTOCOL_NAME = "full_expanded"
RUN_MULTI_MODEL = False
```

## Что смотреть после прогона

### Пер-сэмпл отчёт
`per_sample_predictions.csv`

Полезные поля:
- `has_valid_json`
- `parse_error`
- `requested_property_keys`
- `few_shot_k`
- `few_shot_demo_ids`

### Пер-property отчёт
`property_metrics.csv`

Основные столбцы:
- `coverage_pct`
- `accuracy_pct`
- `macro_f1_pct`
- `selective_accuracy_pct`
- `coverage_on_gt_known_pct`

### Summary
`summary.json`

Полезно смотреть:
- `valid_json_pct`
- `image_id_match_pct`
- `mean_coverage_pct`
- `mean_accuracy_pct`
- `mean_macro_f1_pct`
- `mean_selective_accuracy_pct`

## Практические рекомендации

### Если модель разваливает JSON
- смотреть `parse_error`
- смотреть `raw_output`
- уменьшать `MAX_SAMPLES` для отладки

### Если модель слишком часто отвечает `unknown`
- сравнить zero-shot и few-shot
- смотреть `coverage_pct`, а не только `accuracy_pct`
- проверить, не включён ли `include_only_gt_known=False` на слишком широком протоколе

### Если запуск слишком медленный
- оставить один `variant`
- начать с `MAX_SAMPLES = 20`
- использовать `FEW_SHOT_SELECTION_MODE = "fixed"`
- подобрать `PROPERTY_BATCH_SIZE = 8, 12, 16`

## Минимальный smoke-check
Перед дорогим полным прогоном полезно:
- поставить `MAX_SAMPLES = 5`
- прогнать один `SELECTED_MODEL`
- убедиться, что отчёты сохраняются и Comet логируется

Локальный аналог для быстрой проверки:
- [scripts/run_abo150_smoke.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_abo150_smoke.py)

## Связанные документы
- [validation_pipeline.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/validation_pipeline.md)
- [archive_review/project_notes](/home/alexander/Projects/VLM-2D-Physics-Boundaries/archive_review/project_notes)
