# Manual Segmentation Review

## Что это
Локальный GUI-инструмент для ручного отбора сегментаций на датасете
[abo_physics_natural_bg_v2](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_physics_natural_bg_v2).

Инструмент показывает:
- оригинальное изображение;
- несколько версий сегментации рядом;
- для каждой версии:
  - preview / overlay;
  - masked-вариант;
  - источник маски.

После этого можно:
- выбрать лучший вариант маски;
- одобрить изображение;
- отклонить изображение;
- пропустить и вернуться позже.

## Скрипт
[review_segmentation_masks.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/review_segmentation_masks.py)

## Что именно показывается
По умолчанию скрипт смотрит на 3 варианта:
- `mask_path`
- `mask_path_hint`
- `mask_path_hint_title`

Для них используются preview-поля:
- `seg_preview_path`
- `seg_preview_path_hint`
- `seg_preview_path_hint_title`

Если для базового варианта нет `seg_preview_path`, используется fallback на `sam_preview_path`.

## Запуск
```bash
python scripts/review_segmentation_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2
```

Если нужен другой output-каталог:
```bash
python scripts/review_segmentation_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --output-dir dataset/abo_physics_natural_bg_v2/review_outputs
```

Если нужно начать с конкретного `image_id`:
```bash
python scripts/review_segmentation_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --start-image-id 51-ZzTmzyPL
```

## Второй проход по `review_outputs`
Если нужно ещё раз пересмотреть уже отобранный subset, можно подать
в скрипт предыдущий `approved_meta.json` как новый `meta-path` и сохранить
результат под новым `review-name`.

Пример:
```bash
python scripts/review_segmentation_masks.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --meta-path dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_approved_meta.json \
  --output-dir dataset/abo_physics_natural_bg_v2/review_outputs \
  --review-name segmentation_review_pass2
```

На выходе появятся:
- `segmentation_review_pass2_decisions.json`
- `segmentation_review_pass2_approved_meta.json`
- `segmentation_review_pass2_approved_ids.txt`

Именно `segmentation_review_pass2_approved_ids.txt` удобно потом использовать
для фильтрации CSV-таблиц по `image_id`.

## Фильтр уже одобренного subset
Если не нужно заново выбирать маску, а нужно просто пройтись по уже
одобренному subset и решать `оставить / удалить`, используй:

```bash
python scripts/filter_review_subset.py \
  --dataset-dir dataset/abo_physics_natural_bg_v2 \
  --meta-path dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_approved_meta.json \
  --output-dir dataset/abo_physics_natural_bg_v2/review_outputs \
  --review-name segmentation_review_final
```

Хоткеи:
- `K` — оставить
- `D` — удалить
- `S` — пропустить
- `←`, `→` — назад / вперёд

На выходе появятся:
- `segmentation_review_final_decisions.json`
- `segmentation_review_final_kept_meta.json`
- `segmentation_review_final_kept_ids.txt`

Именно `segmentation_review_final_kept_ids.txt` потом удобно использовать
для фильтрации итоговых таблиц по `image_id`.

## Хоткеи
- `1`, `2`, `3` — выбрать вариант маски
- `A` — одобрить выбранный вариант
- `R` — отклонить
- `S` — пропустить
- `←`, `→` — назад / вперёд

## Что сохраняется
В output-папку пишутся:
- `segmentation_review_decisions.json`
- `segmentation_review_approved_meta.json`
- `segmentation_review_approved_ids.txt`

## Что лежит в `approved_meta`
Для одобренных объектов:
- сохраняется полная исходная metadata-запись;
- канонический `mask_path` переписывается на выбранную вручную версию;
- `mask_source` тоже переписывается на выбранный источник;
- добавляются review-поля:
  - `review_decision`
  - `review_selected_mask_field`
  - `review_selected_mask_source`
  - `review_selected_preview_path`
  - `review_timestamp`

Это сделано специально, чтобы потом можно было использовать
`approved_meta.json` как обычный meta-файл для следующих прогонов.

## Замечание
Скрипт использует `tkinter`. Если в системе его нет, нужно установить системный пакет для Tk:
- например `python3-tk` на Ubuntu / Debian.
