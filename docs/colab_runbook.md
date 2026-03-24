# Colab Runbook

## Что это
Короткая инструкция по запуску основного validation pipeline в Google Colab.

Основной ноутбук:
[ABO150_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/ABO150_Validation_Colab.ipynb)

## Перед запуском

### 1. Colab Secrets
Если нужен приватный репозиторий и Comet, в `google.colab.userdata` должны быть:
- `git_coursework`
- `comet_api_key`
- `comet_workspace`
- `comet_project_name`

Минимально для запуска приватного repo нужен `git_coursework`.

### 2. Runtime
Рекомендуется GPU runtime.

Для моделей:
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`

лучше ориентироваться на Colab с достаточно большой VRAM.

## Основные настройки в ноутбуке

### Протокол
```python
PROTOCOL_NAME = "pdf_compact"
```

Рекомендуемое значение для baseline по coursework:
- `pdf_compact`

### Prompt-режим
```python
PROMPT_MODE = "per_property"
```

Рекомендуемый стартовый режим:
- `per_property`

### Zero-shot / few-shot
```python
FEW_SHOT_K = 0
```

Рекомендуемые значения:
- `0` для zero-shot
- `1` или `2` для few-shot baseline

### Визуальные варианты
```python
EVAL_VARIANTS = ["raw", "mask_overlay", "masked"]
```

Если masks отсутствуют, реально отработает только `raw`.

### Ограничение длины prompt
```python
MAX_PROPERTIES_PER_SAMPLE = None
```

Если используешь `joint`, лучше ограничивать:
- `24` для `pdf_compact`
- `32` для `expanded_ontology`

## Рекомендуемый порядок прогонов

### Первый baseline
```python
PROTOCOL_NAME = "pdf_compact"
PROMPT_MODE = "per_property"
FEW_SHOT_K = 0
RUN_MULTI_MODEL = False
SELECTED_MODEL = "qwen3_vl_8b"
```

### Первый few-shot baseline
```python
PROTOCOL_NAME = "pdf_compact"
PROMPT_MODE = "per_property"
FEW_SHOT_K = 2
RUN_MULTI_MODEL = False
SELECTED_MODEL = "qwen3_vl_8b"
```

### Сравнение нескольких моделей
```python
RUN_MULTI_MODEL = True
MULTI_MODEL_KEYS = ["qwen3_vl_8b", "qwen2_5_vl_7b", "llava_onevision_1_5_8b"]
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
- перейти на `per_property`
- уменьшить `MAX_PROPERTIES_PER_SAMPLE`
- проверить `raw_output`

### Если модель слишком часто отвечает `unknown`
- сравнить zero-shot и few-shot
- сравнить `raw` и `mask_overlay`
- смотреть `coverage_pct`, а не только `accuracy_pct`

### Если запуск слишком медленный
- для first pass оставить один `variant`
- начать с `MAX_SAMPLES = 20`
- помнить, что few-shot `per_property` медленнее zero-shot

## Минимальный smoke-check
Перед дорогим полным прогоном полезно:
- поставить `MAX_SAMPLES = 5`
- прогнать один `SELECTED_MODEL`
- убедиться, что отчёты сохраняются и Comet логируется

## Связанные документы
- [docs/validation_pipeline.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/validation_pipeline.md)
- [day1_requirements.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day1_requirements.md)
- [day2_report.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day2_report.md)
