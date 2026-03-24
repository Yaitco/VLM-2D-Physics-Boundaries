# Validation Pipeline

## Назначение
Этот пайплайн нужен для валидации VLM на задаче извлечения физических свойств объектов из изображений датасета `dataset/abo_150_expanded`.

Основной код лежит в [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py).

## Что поддерживается сейчас

### Протоколы свойств
- `expanded_ontology`
  Использует полную ontology из `dataset/abo_150_expanded/physics_properties.yaml`.

- `pdf_compact`
  Использует компактный набор свойств из PDF coursework.
  Конфиг: [configs/pdf_protocol_properties.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/pdf_protocol_properties.yaml)

### Модели
Сейчас в общем registry поддерживаются:
- `qwen3_vl_8b`
- `qwen2_5_vl_7b`
- `llava_onevision_1_5_8b`

### Визуальные варианты
- `raw`
  Исходная панель объекта.

- `mask_overlay`
  Исходное изображение с подсветкой маски объекта.

- `masked`
  Объект на нейтральном фоне по маске.

`mask_overlay` и `masked` доступны только если в metadata есть `mask_path`.

### Prompt-режимы
- `joint`
  Один запрос на изображение, в котором сразу перечислены несколько свойств.

- `per_property`
  Отдельный запрос на каждое свойство.

### Shot-режимы
- `few_shot_k = 0`
  Zero-shot.

- `few_shot_k > 0`
  Few-shot с демонстрационными примерами из датасета.

## Общая схема работы

### 1. Загрузка схемы свойств
Сначала выбирается evaluation protocol:
- `load_protocol_property_specs("expanded_ontology", ...)`
- `load_protocol_property_specs("pdf_compact", ...)`

На выходе получаем словарь `property_specs`.

### 2. Загрузка samples
Функция `load_abo150_samples(...)`:
- читает `selected_150_annotations.jsonl`
- находит panel image
- подтягивает `mask_path`, если он есть
- формирует `gt_properties`

Для `pdf_compact` GT сначала маппится из ontology ABO в компактные категории.

### 3. Выбор properties для конкретного sample
Функция `select_property_keys_for_sample(...)`:
- либо выбирает только свойства с известным GT
- либо берёт все свойства протокола

### 4. Построение prompt
- `build_prompt_for_sample(...)` для `joint`
- `build_single_property_prompt(...)` для `per_property`

Если включён few-shot, перед основным запросом добавляются demo-примеры:
- demo image
- prompt в том же формате
- assistant-ответ с эталонным JSON из GT

### 5. Inference
Общий runtime интерфейс:
- `infer_runtime(...)`
- `infer_runtime_batch(...)`
- `infer_runtime_messages(...)`

Логика:
- zero-shot `per_property` использует batch inference
- few-shot `per_property` идёт последовательно, потому что у каждого свойства свой набор demo-изображений

### 6. Парсинг и нормализация
После ответа модели:
- извлекается JSON
- значения нормализуются к allowed enum
- unknown/empty приводятся к единому формату

### 7. Подсчёт метрик
На уровне sample считаются:
- `has_valid_json`
- `image_id_matched`
- `pred_known`
- `missed_when_gt_known`
- `exact_match`

На уровне property считаются:
- `coverage_pct`
- `accuracy_pct`
- `macro_f1_pct`
- `selective_accuracy_pct`
- `coverage_on_gt_known_pct`
- `exact_match_on_gt_known_pct`

### 8. Сохранение отчётов
Для каждого `model_key / variant` сохраняются:
- `per_sample_predictions.csv`
- `property_metrics.csv`
- `summary.json`

## Few-shot: текущая реализация

### Как выбираются demo-примеры
Используется `select_few_shot_examples(...)`.

Принцип:
- текущий sample исключается
- кандидатам считается score по числу known GT
- выбираются top-k примеров с максимальным score

Для `per_property` score считается по одному свойству.
Для `joint` score считается по всему списку выбранных свойств.

### Что важно понимать
- Few-shot сейчас строится из самого evaluation subset.
- Это удобно для controlled baseline, но методологически это нужно отдельно оговорить в отчёте.
- Если понадобится жёсткое разделение `support set` и `evaluation set`, это можно добавить отдельным следующим шагом.

## Основные параметры запуска

Ключевые аргументы `run_validation(...)`:
- `model_key`
- `samples`
- `property_specs`
- `variant`
- `prompt_mode`
- `property_batch_size`
- `include_only_gt_known`
- `few_shot_k`
- `max_properties_per_sample`
- `mask_background_mode`
- `json_success_threshold`
- `image_id_success_threshold`

## Ограничения
- `wetness` в compact-протоколе сейчас почти всегда `unknown`, потому что в доступной разметке нет устойчивого прямого поля.
- `temperature_hint`, `phase`, часть `filled_state` слабо вариативны в текущем subset.
- Few-shot в `per_property` режиме медленнее zero-shot, потому что отключается batch generate.
- Для real multi-model run модели по-прежнему запускаются последовательно, а не параллельно.

## Что смотреть при отладке

### Если `has_valid_json=False`
Смотреть:
- `parse_error`
- `raw_output`
- `valid_json_ratio`

### Если модель почти везде отвечает `unknown`
Смотреть:
- `pred_yes_pct`
- `coverage_pct`
- `selective_accuracy_pct`

### Если есть сомнения в GT compact-протокола
Смотреть mapping-функции:
- `_map_pdf_material`
- `_map_pdf_reflectance`
- `_map_pdf_surface_roughness`
- `_map_pdf_rigidity`
- `_map_pdf_fragility`
- `_map_pdf_state`
- `_map_pdf_weight_hint`
- `_map_pdf_temperature_hint`
- `_map_pdf_phase`
- `_map_pdf_filled_state`
- `_map_pdf_slipperiness_hint`

## Связанные файлы
- [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py)
- [ABO150_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/ABO150_Validation_Colab.ipynb)
- [configs/pdf_protocol_properties.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/pdf_protocol_properties.yaml)
- [scripts/run_abo150_smoke.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_abo150_smoke.py)
