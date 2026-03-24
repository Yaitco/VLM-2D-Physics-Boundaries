# Отчёт по Дню 2

## Цель дня
Довести пайплайн до состояния, в котором можно запускать baseline-эксперименты по протоколу из `Coursework.pdf`, а не только по расширенной ontology датасета.

## Что было сделано

### 1. Добавлен компактный evaluation-протокол из PDF
- Создан файл конфигурации: [configs/pdf_protocol_properties.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/pdf_protocol_properties.yaml)
- В протокол включены 13 свойств:
  - `material`
  - `transparency`
  - `reflectance`
  - `surface_roughness`
  - `rigidity`
  - `fragility`
  - `wetness`
  - `state`
  - `weight_hint`
  - `temperature_hint`
  - `phase`
  - `filled_state`
  - `slipperiness_hint`

### 2. Добавено переключение между двумя протоколами
В [scripts/abo150_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/abo150_vlm_validation.py) реализована загрузка двух режимов:
- `expanded_ontology`
- `pdf_compact`

Это позволяет:
- запускать прежнюю расширенную валидацию без поломок;
- отдельно запускать основной протокол, соответствующий заданию из PDF.

### 3. Реализован mapping GT из ABO в компактный PDF-протокол
Для `pdf_compact` добавено преобразование полей ABO-разметки в компактные категории.

Примеры:
- `main_material -> material`
- `glossiness_class -> reflectance`
- `rigidity_class -> rigidity`
- `breakable / brittleness_class -> fragility`
- `mass_class -> weight_hint`
- `object_temperature_class -> temperature_hint`
- `state_of_matter -> phase`
- `fill_state_class -> filled_state`
- `friction_class -> slipperiness_hint`

Это закрывает главный разрыв между текущим датасетом и формулировкой задачи в PDF.

### 4. Добавлены метрики, нужные для Day 2 baseline
В агрегированный отчёт `property_metrics.csv` добавлены:
- `coverage_pct`
- `accuracy_pct`
- `macro_f1_pct`
- `selective_accuracy_pct`

Также summary-отчёт теперь содержит:
- `mean_coverage_pct`
- `mean_accuracy_pct`
- `mean_macro_f1_pct`
- `mean_selective_accuracy_pct`

Это делает результаты пригодными для дальнейших сравнений между моделями, prompt-режимами и визуальными вариантами.

### 5. Добавлен новый визуальный режим `mask_overlay`
В пайплайн добаван режим:
- `raw`
- `mask_overlay`
- `masked`

`mask_overlay` сохраняет фон, но подсвечивает объект по маске. Это важно для будущего эксперимента про влияние сегментации на качество извлечения свойств.

### 6. Обновлён Colab-ноутбук
Обновлён [ABO150_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/ABO150_Validation_Colab.ipynb):
- добавлен `PROTOCOL_NAME = "pdf_compact"`
- переключение на `load_protocol_property_specs(...)`
- загрузка samples с `protocol_name=PROTOCOL_NAME`
- список вариантов обновлён до `["raw", "mask_overlay", "masked"]`
- отчёты сохраняются по подпапкам протокола

Итог: baseline теперь можно запускать из Colab без ручного редактирования Python-скриптов.

### 7. Добавлен и проверен smoke-run для Day 2
Обновлён [scripts/run_abo150_smoke.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_abo150_smoke.py):
- поддерживает `--protocol-name pdf_compact`
- проходит полный цикл:
  - загрузка датасета
  - построение GT
  - validation loop
  - сохранение отчётов

## Что проверено

### Синтаксис
Проверка прошла успешно:
```bash
python -m py_compile scripts/abo150_vlm_validation.py scripts/run_abo150_smoke.py
```

### Smoke-run на компактном протоколе
Проверка прошла успешно:
```bash
python scripts/run_abo150_smoke.py --max-samples 5 --prompt-mode per_property --protocol-name pdf_compact
```

Результат:
- `valid_json_pct = 100.0`
- `image_id_match_pct = 100.0`
- `valid_json_per_response_pct = 100.0`

### Проверка `mask_overlay`
Отдельно протестировано наложение маски на синтетическом примере. Режим работает корректно:
- фон сохраняется;
- объект подсвечивается;
- граница маски выделяется отдельно.

## Найденные проблемы и исправления

### Проблема 1. Compact protocol падал на плоских ключах
Симптом:
- `runtime_error: not enough values to unpack (expected 2, got 1)`

Причина:
- helper для чтения предсказаний ожидал ключи формата `group.name`, а в compact-протоколе ключи плоские: `material`, `state`, `fragility`.

Исправление:
- обновлена логика `_fetch_pred_raw_value(...)`, чтобы она корректно обрабатывала оба случая.

### Проблема 2. Раньше в проекте не было mask-overlay
Причина:
- существовал только `raw` и частично `masked`.

Исправление:
- добавлен новый вариант изображения без разрушения контекста сцены.

## Ограничения на конец Дня 2

- Few-shot режим ещё не реализован.
- Реальные baseline-прогоны на GPU-моделях в этом отчёте не запускались из текущей локальной среды.
- `mask_overlay` и `masked` будут доступны только если у записей реально есть `mask_path`.
- Некоторые compact-свойства в текущем датасете будут сильно несбалансированы или почти всегда `unknown`:
  - `wetness`
  - `temperature_hint`
  - `phase`
  - часть `filled_state`

Это не ошибка пайплайна, а ограничение доступной разметки и состава подмножества.

## Вывод дня
День 2 закрыл инженерный минимум для baseline-экспериментов по формулировке из PDF:
- есть компактный протокол;
- есть нужные метрики;
- есть режимы `raw` / `mask_overlay` / `masked`;
- есть Colab-точка входа;
- есть smoke-проверка на полном цикле.

На практике это означает, что следующий шаг уже должен быть не про инфраструктуру, а про реальные эксперименты:
- zero-shot baseline,
- few-shot режим,
- сравнение по моделям и визуальным вариантам.

## Следующий шаг
Приоритет на следующий рабочий блок:
1. добавить few-shot skeleton в общий pipeline;
2. запустить первый реальный zero-shot baseline в Colab на `pdf_compact`;
3. сохранить первые таблицы результатов по моделям и свойствам.
