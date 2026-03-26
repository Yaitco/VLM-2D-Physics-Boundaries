# VLM-2D-Physics-Boundaries

Репозиторий приведён к более чистой структуре вокруг одного общего validation pipeline.

## Активная структура
- `notebooks/`
  - Colab и рабочие ноутбуки
- `scripts/`
  - CLI-раннеры и утилиты подготовки датасета
- `vlm_pipeline/`
  - реестры и ядро pipeline
- `configs/`
  - конфиги протоколов и компактных схем
- `dataset/`
  - датасеты и производные subset'ы
- `docs/`
  - runbook и техническая документация
- `archive_review/`
  - дубли, старые входы, промежуточные заметки и артефакты на ревью

## Рекомендуемый вход
- Colab: [Unified_VLM_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/notebooks/Unified_VLM_Validation_Colab.ipynb)
- CLI: [run_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_vlm_validation.py)

## Где что менять

### Добавить новую модель
- редактировать [registry.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/registry.py)
- добавить запись в `MODEL_REGISTRY`

### Добавить новый датасет
- редактировать [registry.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/registry.py)
- добавить запись в `get_dataset_context(...)`
- при необходимости добавить новый loader в [datasets.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/datasets.py)
- если нужен отдельный protocol mapping, добавить его в [specs.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/specs.py)

### Добавить новый компактный протокол
- добавить YAML в [configs](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs)
- подключить его в [specs.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/specs.py)

### Куда смотреть по слоям
- [registry.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/registry.py)
  - модели и dataset contexts
- [specs.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/specs.py)
  - схемы свойств, протоколы и нормализация значений
- [datasets.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/datasets.py)
  - загрузка samples и GT
- [runtime.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/runtime.py)
  - загрузка моделей и inference backends
- [images.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/images.py)
  - `raw`, `mask_overlay`, `masked`
- [evaluation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/evaluation.py)
  - prompts, few-shot и валидационный цикл
- [reporting.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/reporting.py)
  - property metrics, `summary.json`, Comet

## Основные документы
- Runbook: [colab_runbook.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/colab_runbook.md)
- Validation pipeline: [validation_pipeline.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/validation_pipeline.md)
- ABO natural background subset: [abo_natural_background_subset.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/abo_natural_background_subset.md)
- SAM masks: [sam_mask_generation.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/sam_mask_generation.md)

## Архив на просмотр
- Старые ноутбуки и специализированные входы перенесены в [archive_review](/home/alexander/Projects/VLM-2D-Physics-Boundaries/archive_review)
