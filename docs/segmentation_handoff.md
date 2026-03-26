# Segmentation Handoff: `abo_physics_natural_bg_v2`

## Что важно знать сразу
- Для этой задачи целевой датасет: **`dataset/abo_physics_natural_bg_v2`**.
- В нём сейчас `220` объектов.
- Он уже подключён в единый validation pipeline.
- Все рекомендации ниже относятся именно к `abo_physics_natural_bg_v2`.

## Где лежат данные
- Датасет: [dataset/abo_physics_natural_bg_v2](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_physics_natural_bg_v2)
- Метаданные: [meta.json](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_physics_natural_bg_v2/meta.json)
- Картинки: `dataset/abo_physics_natural_bg_v2/images/...`
- Маски: `dataset/abo_physics_natural_bg_v2/masks/...`

## Формат одного sample в `meta.json`
Ключи, которые реально есть у объекта:
- `image_id`
- `path`
- `mask_path`
- `mask_source`
- `primary_object`
- `notes`
- `properties`
- `abo_meta`
- `background_metrics`

### Что важно для сегментации
- `path`
  - путь до изображения
  - хранится как относительный путь вида `abo_physics_natural_bg_v2/images/13/13b61d5c.jpg`
- `mask_path`
  - текущая маска
  - сейчас это базовая маска в `dataset/abo_physics_natural_bg_v2/masks/...`
- `mask_source`
  - откуда эта маска взялась
- `primary_object`
  - короткое описание главного объекта
  - можно использовать как text prompt, если делать grounded segmentation
- `abo_meta.title`
  - текстовое название товара
- `abo_meta.product_type`
  - тип товара
- `abo_meta.domain_name`
  - домен/категория

Пример полезных текстовых источников:
- `primary_object`: лучшее короткое описание объекта
- `abo_meta.product_type`: более грубый класс
- `abo_meta.title`: более длинное, но иногда шумное описание

## Как текущая валидация использует маски
Маска подхватывается автоматически из `mask_path`.

Файл: [images.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/images.py)

Логика такая:
- `raw`
  - берётся исходное изображение из `path`
- `masked`
  - всё вне маски из `mask_path` закрашивается в чёрный или белый фон
- `mask_overlay`
  - исходное изображение остаётся, поверх маски рисуется полупрозрачный overlay

### Практический вывод
Если хочется, чтобы эксперименты `raw vs mask_overlay vs masked` сразу работали на новой сегментации, самый простой контракт такой:
- обновлять `mask_path` в `meta.json` на новую маску;
- обновлять `mask_source`, например на `sam:facebook/sam-vit-base` или `grounded_sam:<...>`.

Тогда validation pipeline ничего не нужно менять.

## Что уже есть по сегментации
Есть базовый скрипт:
- [generate_sam_masks.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/generate_sam_masks.py)

Документация:
- [sam_mask_generation.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/sam_mask_generation.md)

### Что он делает сейчас
Текущий baseline работает так:
1. Берёт текущую `mask_path` как слабую seed-маску.
2. Строит по seed-маске `bbox`.
3. Передаёт `bbox` в SAM.
4. Выбирает лучшую SAM-маску по смеси:
   - overlap с seed-маской;
   - `predicted_iou` от SAM;
   - близость по площади к seed.

### Что он пишет обратно в `meta.json`
- `sam_seed_mask_path`
- `sam_seed_mask_source`
- `sam_prompt_mask_field`
- `sam_prompt_source_field`
- `sam_prompt_box_xyxy`
- `sam_predicted_iou`
- `sam_overlap_with_seed_iou`
- `sam_mask_area_ratio`
- `sam_status`
- `sam_preview_path` при `--write-previews`

### Что он сохраняет дополнительно
- backup метаданных: `meta.before_sam.json`
- summary: `sam_summary.json`
- превью: `masks_preview/...` при `--write-previews`

## Базовый запуск текущего SAM baseline
```bash
python scripts/generate_sam_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --sam-model-id facebook/sam-vit-base \
  --device cuda \
  --save-every 10 \
  --write-previews
```

## Что именно нужно сделать под новую задачу
Идейно задача выглядит так:

### Режим A: модель не знает, что ищет
Цель:
- найти “главный объект” без текстового названия класса;
- потом сегментировать его через SAM.

Что можно пробовать:
- saliency / main-object detector;
- VLM/детектор, который даёт point(s) или box главного объекта;
- затем SAM по positive points и, возможно, negative points.

### Режим B: модель знает, что ищет
Цель:
- искать конкретный объект по metadata и сегментировать именно его.

Источники текста:
1. `primary_object`
2. `abo_meta.product_type`
3. `abo_meta.title`

Режимы сравнения, которые реально интересны:
- `unknown_target`: модель ищет просто main object
- `known_target_primary_object`: модель знает `primary_object`
- `known_target_product_type`: модель знает `product_type`
- `known_target_title`: модель знает `title`

## Что лучше сохранять для экспериментов
Для главы и качественного анализа желательно заранее сохранять артефакты, а не генерировать их потом на лету.

Минимальный набор на каждый sample:
- raw image
- predicted box / points visualization
- final binary mask
- overlay preview
- masked object image

### Что ещё полезно сохранять в `meta.json`
Если делается новый segmentation pipeline, хорошо добавить поля:
- `seg_prompt_mode`
  - например `main_object`, `primary_object`, `product_type`, `title`
- `seg_query_text`
  - какой текст реально ушёл в grounding/model
- `seg_box_xyxy`
- `seg_positive_points`
- `seg_negative_points`
- `seg_model_name`
- `seg_status`
- `seg_error`
- `seg_mask_area_ratio`

Если хочется не терять старую маску, можно ещё хранить:
- `seed_mask_path`
- `seed_mask_source`

## Самый безопасный контракт с текущим репозиторием
Чтобы ничего не сломать в основном pipeline:
- итоговую лучшую маску писать в `mask_path`;
- источник писать в `mask_source`;
- дополнительные поля писать рядом как новые `seg_*` или `sam_*`;
- пути хранить так же, как сейчас: относительно `dataset/`, например
  `abo_physics_natural_bg_v2/masks/13/13b61d5c.png`.

## Как потом прогнать влияние сегментации
Единый раннер:
- [run_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_vlm_validation.py)

Пример:
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_physics_natural_bg_v2 \
  --protocol-name natural_bg_v2 \
  --model-key qwen2_5_vl_7b \
  --variants raw,mask_overlay,masked \
  --property-batch-size 8
```

Это сравнит:
- `raw`
- `mask_overlay`
- `masked`

При условии, что в `meta.json` корректно обновлён `mask_path`.

## Какой протокол сейчас используется для `v2`
Файл:
- [abo_natural_bg_v2_properties.yaml](/home/alexander/Projects/VLM-2D-Physics-Boundaries/configs/abo_natural_bg_v2_properties.yaml)

Свойства:
- `material`
- `rigidity`
- `transparency`
- `surface`
- `fragility`

## Практическая рекомендация по реализации
Если делать новую систему сегментации, я бы шёл так:
1. Отдельный новый скрипт, не переписывая сразу старый `generate_sam_masks.py`.
2. Сначала debug-режим на 10–20 изображениях.
3. Обязательно сохранять preview-артефакты.
4. Сначала сравнить:
   - без знания объекта
   - с `primary_object`
5. Только потом добавлять `title` и более сложные negative prompts.

## На что обратить внимание
- На некоторых изображениях есть посторонний объект или животное на товаре.
- В таких кейсах text-aware режим скорее всего будет важнее, чем просто SAM от seed-маски.
- Если задача именно “найти ковёр, а не собаку на ковре”, то plain SAM без хорошего prompt'а обычно недостаточен.

## Если нужен минимум, который должен получиться на выходе
- новая маска в `mask_path`
- `mask_source`
- preview-картинки
- summary по качеству/статусам
- возможность сразу прогнать `raw vs mask_overlay vs masked` без правок валидатора
