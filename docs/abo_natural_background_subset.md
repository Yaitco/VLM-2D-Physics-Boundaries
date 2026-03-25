# ABO Natural Background Subset

## Что это
Скрипт [build_abo_physics_subset.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/build_abo_physics_subset.py) теперь умеет фильтровать ABO-кандидатов по признакам естественного фона, а не только по “не белому” бордеру.

Основная идея:
- сначала собирается пул кандидатов из ABO listing metadata;
- затем материализуются изображения;
- для каждого изображения считаются метрики border-region и оценённой background-region;
- в subset попадают только изображения с достаточно богатым не-studio контекстом.

## Какие признаки фона считаются
Для border-области считаются:
- `border_white_ratio`
- `border_dark_ratio`
- `border_std`
- `border_saturation`
- `border_neutral_ratio`
- `border_dominant_bin_ratio`
- `border_color_entropy`
- `context_score`

Дополнительно для оценённой связной background-области считаются:
- `background_area_ratio`
- `background_white_ratio`
- `background_dark_ratio`
- `background_std`
- `background_saturation`
- `background_neutral_ratio`
- `background_dominant_bin_ratio`
- `background_color_entropy`

Интуиция:
- однотонный белый или чёрный фон даёт высокий `dominant_bin_ratio` и низкий `context_score`;
- более естественная сцена даёт более высокий `border_std`, `border_color_entropy` и обычно более низкий `dominant_bin_ratio`.
- связная background-область помогает отдельно штрафовать ровный white/black studio backdrop.

## Рекомендуемый стартовый запуск
```bash
python scripts/build_abo_physics_subset.py \
  --output-name abo_physics_natural_bg \
  --listing-shards 0,1,2,3 \
  --max-samples 240 \
  --selection-pool-multiplier 8 \
  --min-known-properties 4 \
  --max-per-product-type 20 \
  --download-missing \
  --generate-masks \
  --mask-backend rembg \
  --min-context-score 0.22 \
  --max-border-white-ratio 0.98 \
  --max-border-dark-ratio 0.98 \
  --max-border-dominant-bin-ratio 0.75 \
  --min-background-area-ratio 0.03 \
  --max-background-white-ratio 0.97 \
  --max-background-dark-ratio 0.97 \
  --max-background-dominant-bin-ratio 0.80
```

## Что смотреть после сборки
Файлы:
- `dataset/abo_physics_natural_bg/meta.json`
- `dataset/abo_physics_natural_bg/summary.json`

Во время работы скрипт теперь печатает прогресс по фазам:
- подготовка metadata;
- загрузка image index;
- построение candidate pool;
- materialization и context filtering;
- генерация масок.

Частоту можно настроить:
- `--progress-every-records 5000`
- `--progress-every-selected 25`

Ключевые поля в `summary.json`:
- `num_candidates`
- `num_selected_before_images`
- `num_written`
- `context_stats`
- `context_score_avg`
- `context_score_min`
- `context_score_max`

Если датасет получился слишком маленьким:
- уменьшить `--min-context-score`
- увеличить `--max-border-dominant-bin-ratio`
- увеличить `--max-background-dominant-bin-ratio`
- увеличить `--selection-pool-multiplier`
- добавить больше `--listing-shards`

## Что сохраняется в sample
В `meta.json` у каждого sample теперь есть:
- `background_metrics`

Это полезно для:
- последующего анализа,
- отладки порогов,
- отбора кандидатов под SAM и эксперименты `raw vs segmented`.
