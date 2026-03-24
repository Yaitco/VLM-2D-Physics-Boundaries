## День 1 — Экспериментальная матрица

### Основной протокол этапа 1
- Датасет: `dataset/abo_150_expanded`
- Основной сценарий: object-level prediction
- Основной набор свойств для отчёта: компактный словарь из PDF
- Дополнительный сценарий: expanded ontology как exploratory appendix / дополнительная абляция

### Модели
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`

### Режимы инференса
- `zero-shot`
- `few-shot (k = 1, 2, 4)`

### Режимы визуального входа
- `raw`
- `mask-overlay`
- `masked-object`

### Prompt-режимы
- `joint`
- `per_property`

### Основные метрики
- `accuracy`
- `macro-F1`
- `coverage`
- `selective accuracy`

### Дополнительные диагностические метрики
- `valid_json_pct`
- `image_id_match_pct`
- `parse_error` distribution
- доля `unknown` в ответах

### Приоритетная очередность экспериментов
1. Zero-shot, `raw`, все модели.
2. Zero-shot, сравнение `raw / mask-overlay / masked-object`.
3. Few-shot на лучшей и средней модели.
4. Joint vs per-property.
5. Лёгкий пилот дообучения одной модели, только если всё выше уже стабильно.

### Что считать сильным результатом
- Есть baseline по всем моделям в одном протоколе.
- Есть сравнение zero-shot и few-shot.
- Есть сравнение режимов сегментации.
- Есть property-level анализ, а не только одна усреднённая цифра.
- Есть содержательный разбор того, где `unknown` — корректный и полезный ответ.

### Что считать допустимым сокращением объёма
- Если few-shot или дообучение не взлетают быстро, не жертвовать ради них zero-shot baseline и segmentation study.
- Если широкая ontology мешает скорости экспериментов, основной отчёт строить на компактном словаре PDF.

## Чекпоинт Дня 1
- После Дня 1 должен существовать один воспроизводимый smoke-run, который проходит путь:
  загрузка датасета -> выбор конфигурации -> запуск validation loop -> сохранение `per_sample_predictions.csv`, `property_metrics.csv`, `summary.json`
