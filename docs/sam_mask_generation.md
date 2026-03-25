# SAM Mask Generation

Этот шаг строит SAM-маски поверх уже собранного subset и по умолчанию сразу заменяет старые `rembg`-маски в `mask_path`.

## Идея

- Берём существующую `mask_path` как слабую seed-маску.
- Строим по ней bbox.
- Передаём bbox в `SAM`.
- Из нескольких SAM-кандидатов выбираем маску с лучшим компромиссом между:
  - overlap с seed-маской;
  - внутренним `predicted_iou` от SAM.

Такой режим практичен для нашего case:
- у нас один главный объект на изображении;
- `rembg` уже даёт грубую локализацию;
- `SAM` уточняет контур.

## Что обновляется

Скрипт:
- по умолчанию сохраняет новые маски в `masks/`, то есть поверх старых `rembg`-файлов;
- обновляет `meta.json`;
- делает backup `meta.before_sam.json`;
- пишет `sam_summary.json`.

Новые поля в `meta.json`:
- `sam_seed_mask_path`
- `sam_seed_mask_source`
- `sam_prompt_mask_field`
- `sam_prompt_source_field`
- `sam_prompt_box_xyxy`
- `sam_predicted_iou`
- `sam_overlap_with_seed_iou`
- `sam_mask_area_ratio`
- `sam_status`

## Базовый запуск

```bash
python scripts/generate_sam_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --sam-model-id facebook/sam-vit-base \
  --device cuda \
  --save-every 10
```

## Отладочный запуск

```bash
python scripts/generate_sam_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --sam-model-id facebook/sam-vit-base \
  --device cuda \
  --max-samples 10 \
  --save-every 2 \
  --write-previews
```

## Полезные параметры

- `--input-mask-field mask_path`
  - какую seed-маску использовать как prompt для SAM
- `--output-mask-field mask_path`
  - по умолчанию SAM обновляет основной `mask_path`
- `--output-source-field mask_source`
  - по умолчанию источник тоже обновляется на `sam:<model_id>`
- `--overwrite`
  - пересчитать уже существующие SAM-маски
- `--bbox-margin-ratio 0.08`
  - насколько расширять bbox вокруг seed-маски
- `--min-seed-area-ratio 0.003`
  - отсекает совсем маленькие seed-маски
- `--write-previews`
  - сохраняет overlay-preview рядом для быстрой проверки качества

## Что смотреть после запуска

- `dataset/abo_physics_natural_bg_v2/sam_summary.json`
- несколько файлов из `dataset/abo_physics_natural_bg_v2/masks/`
- если включён `--write-previews`, ещё и `masks_preview/`

Если `failures` много, сначала удобно посмотреть:
- `sam_status`
- `sam_error`
- `sam_overlap_with_seed_iou`
- `sam_mask_area_ratio`
