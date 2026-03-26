# vlm_pipeline

Пакет с активным ядром validation pipeline.

## Структура
- [registry.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/registry.py)
  - `MODEL_REGISTRY`
  - `DatasetContext`
  - `get_dataset_context(...)`

- [specs.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/specs.py)
  - загрузка protocol schemas
  - `PropertySpec`
  - нормализация значений и сравнение GT/pred

- [datasets.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/datasets.py)
  - загрузка `abo_150_expanded`
  - загрузка `abo_physics_natural_bg_v2`
  - compact mapping для `pdf_compact`

- [runtime.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/runtime.py)
  - загрузка моделей
  - backend registry
  - batched / multi-image inference

- [images.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/images.py)
  - `raw`
  - `mask_overlay`
  - `masked`

- [parsing.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/parsing.py)
  - извлечение JSON
  - нормализация ответа модели

- [evaluation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/evaluation.py)
  - prompt building
  - few-shot selection
  - per-sample evaluation loop
  - multi-model runner

- [reporting.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/reporting.py)
  - property metrics
  - `summary.json`
  - Comet logging

- [comet.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/comet.py)
  - инициализация Comet и чтение секретов

## Как расширять

### Добавить новую модель
1. Добавить запись в `MODEL_REGISTRY` в `registry.py`
2. Если нужен новый backend, реализовать его в `runtime.py`

### Добавить новый датасет
1. Добавить новый `DatasetContext` в `registry.py`
2. Если формат разметки новый, добавить loader в `datasets.py`

### Добавить новый протокол свойств
1. Положить YAML в `configs/`
2. Подключить его в `specs.py`

## Совместимость
Файл [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py)
реэкспортирует публичный API отсюда, чтобы ноутбуки и старые входы не ломались.
