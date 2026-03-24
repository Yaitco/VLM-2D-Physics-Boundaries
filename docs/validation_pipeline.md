# Validation Pipeline

## Назначение
Репозиторий теперь использует один основной сценарий валидации: `per_property`.

Идея простая:
- для каждого изображения выбирается список свойств протокола;
- на каждое свойство отправляется отдельный короткий prompt;
- ответы нормализуются и сводятся в единый `per_sample_predictions.csv`.

Основной код находится в [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py).

## Что поддерживается

### Протоколы свойств
- `narrow_core`
  Узкий набор свойств для быстрых baseline-сравнений.
  Ключи лежат в [configs/narrow_core_property_keys.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/narrow_core_property_keys.yaml).

- `full_expanded`
  Полная ontology из `dataset/abo_150_expanded/physics_properties.yaml`.

- `expanded_ontology`
  Alias для полного протокола.

- `pdf_compact`
  Компактный маппинг свойств под coursework-протокол.
  Конфиг лежит в [configs/pdf_protocol_properties.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/pdf_protocol_properties.yaml).

### Модели
Сейчас в общем registry доступны:
- `qwen3_vl_8b`
- `qwen2_5_vl_7b`
- `llava_onevision_1_5_8b`

### Визуальные варианты
- `raw`
  Исходная панель объекта.

- `mask_overlay`
  Исходное изображение с наложенной подсветкой маски.

- `masked`
  Объект, вырезанный по маске на однотонный фон.

`mask_overlay` и `masked` доступны только если у sample есть `mask_path`.

### Zero-shot и few-shot
- `few_shot_k = 0`
  Обычный zero-shot.

- `few_shot_k > 0`
  Few-shot с демонстрационными примерами из того же subset.

Поддерживаются два способа выбора demo:
- `fixed`
  Один глобальный ранжированный список кандидатов на весь запуск.

- `dynamic`
  Демонстрации подбираются отдельно под конкретный sample и property.

## Общая схема

### 1. Загрузка схемы свойств
Функция `load_protocol_property_specs(...)` возвращает словарь `property_specs` для выбранного протокола.

### 2. Загрузка датасета
Функция `load_abo150_samples(...)`:
- читает `selected_150_annotations.jsonl`;
- резолвит `panel_path`;
- при наличии подтягивает `mask_path`;
- строит `gt_properties` в формате выбранного протокола.

Для `pdf_compact` GT предварительно маппится из ontology ABO в компактные категории.

### 3. Выбор свойств для sample
Функция `select_property_keys_for_sample(...)`:
- берёт все свойства протокола;
- либо только те, у которых GT известен, если включён `include_only_gt_known=True`.

### 4. Построение prompt
Для каждого свойства строится отдельный короткий prompt:
- image id;
- имя свойства;
- допустимые значения;
- требуемый JSON-формат ответа.

Функция: `build_property_prompt(...)`.

### 5. Few-shot сообщения
Если `few_shot_k > 0`, перед основным запросом добавляются demo-примеры:
- demo image;
- такой же property prompt;
- assistant-ответ с эталонным JSON из GT.

Для fixed few-shot используется общий кэш кандидатов, чтобы ускорить запуск и вернуть batched inference.

### 6. Inference
Инференс идёт через единый runtime layer:
- `load_runtime(...)`
- `infer_runtime_batch(...)`
- `infer_runtime_messages_batch(...)`

Оптимизация сейчас такая:
- zero-shot per-property идёт батчами;
- few-shot с `fixed` тоже старается идти батчами;
- если backend не умеет корректно батчить multi-image conversations, используется безопасный fallback.

### 7. Парсинг и нормализация
После ответа модели:
- вытаскивается JSON;
- значения нормализуются к allowed enum;
- `unknown` и пустые списки приводятся к единому формату.

### 8. Подсчёт метрик
На уровне sample считаются:
- `has_valid_json`
- `has_valid_json_all`
- `image_id_matched`
- `image_id_matched_all`
- `valid_json_ratio`
- `image_id_match_ratio`

На уровне свойства считаются:
- `pred_yes_pct`
- `coverage_pct`
- `coverage_on_gt_known_pct`
- `accuracy_pct`
- `macro_f1_pct`
- `selective_accuracy_pct`
- `exact_match_on_gt_known_pct`

### 9. Сохранение отчётов
Для каждого `model_key / variant` сохраняются:
- `per_sample_predictions.csv`
- `property_metrics.csv`
- `summary.json`

## Основные параметры запуска
Главные аргументы `run_validation(...)`:
- `model_key`
- `samples`
- `property_specs`
- `variant`
- `property_batch_size`
- `include_only_gt_known`
- `few_shot_k`
- `few_shot_selection_mode`
- `mask_background_mode`
- `json_success_threshold`
- `image_id_success_threshold`

## Практические рекомендации

### Быстрый baseline
- `PROTOCOL_NAME = "narrow_core"`
- `EVAL_VARIANTS = ["raw"]`
- `FEW_SHOT_K = 0`
- `PROPERTY_BATCH_SIZE = 8` или `16`

### Более надёжный comparison
- фиксировать `MAX_SAMPLES` и `RANDOM_SEED`;
- сравнивать модели на одном и том же subset;
- менять за раз только одну ось: модель, few-shot или variant.

### Когда идти в полный протокол
Сначала выбрать 1–2 лучшие модели на `narrow_core`, потом запускать их на `full_expanded`.

## Ограничения
- `pdf_compact` использует mapping из ontology ABO, поэтому это не “родной” GT, а производный протокол.
- Few-shot сейчас строится из того же evaluation subset; это нужно явно оговаривать в отчёте.
- Multi-model запуск всё ещё последовательный по моделям.

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

### Если есть сомнения в compact mapping
Смотреть mapping-функции в [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py):
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
- [scripts/run_abo150_smoke.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_abo150_smoke.py)
- [docs/colab_runbook.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/colab_runbook.md)
