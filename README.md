# Grounding DINO + SAM Carpet Segmentation Experiment

Этот проект полностью переписан без `SAM 3`.

Текущий pipeline строго такой:

- `main_object`: `SAM Automatic Mask Generator -> heuristic mask selection`
- `class-specific`: `Grounding DINO (text -> box) -> SAM (box -> mask)`

Поддерживаются 4 метода сегментации:

- `main_object`
- `carpet`
- `rug`
- `area_rug`

## Структура проекта

```text
src/
  models/
    grounding_dino.py
    sam_wrapper.py
  pipelines/
    main_object_pipeline.py
    carpet_pipeline.py
  utils/
    dataset_update.py
    metrics.py
    visualization.py
    postprocess.py
  run_experiment.py
```

## Что делает эксперимент

Для каждого изображения проект:

1. строит `main_object`-маску через `SamAutomaticMaskGenerator`;
2. прогоняет `Grounding DINO` для prompt-ов `carpet`, `rug`, `area rug`;
3. берет лучший bounding box по `dino_score`;
4. передает box в `SAM SamPredictor` и берет лучшую маску по `sam_score`;
5. делает post-processing:
   - удаление маленьких компонент;
   - заполнение дыр;
   - morphological closing;
6. считает метрики при наличии GT:
   - IoU;
   - Dice;
   - Precision;
   - Recall;
7. сохраняет:
   - бинарные маски;
   - overlay-визуализации;
   - коллажи;
   - CSV-таблицы с результатами и аналитикой.

Дополнительно проект умеет обновлять датасет in-place:

- писать выбранную маску в `dataset/.../masks/...`;
- обновлять `mask_path` и `mask_source` в `meta.json`;
- сохранять backup `meta.before_grounded_sam.json`;
- писать `grounded_sam_summary.json`;
- при желании сохранять preview в `masks_preview/...`.

## Установка

Создай отдельное окружение и установи зависимости:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если установка `segment-anything` через `git+https` зависает, установи его вручную:

```bash
git clone https://github.com/facebookresearch/segment-anything.git
pip install -e ./segment-anything
```

## Модели

### Grounding DINO

В проекте используется `Grounding DINO` через `transformers`, по умолчанию:

- `IDEA-Research/grounding-dino-base`

Модель скачивается автоматически при первом запуске. При желании можно передать локальную директорию вместо model id через `--grounding_model_id`.

### SAM

Нужно заранее скачать checkpoint SAM и либо:

- положить его в `checkpoints/sam_vit_b_01ec64.pth`, либо
- передать путь через `--sam_checkpoint`.

Пример для `vit_b`:

- файл: `sam_vit_b_01ec64.pth`
- аргумент: `--sam_model_type vit_b`

## Формат данных

### Изображения

Поддерживаются:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`
- `.tif`
- `.tiff`

### Ground truth

Предполагается, что GT-маска имеет тот же stem, что и изображение.

Пример:

- image: `sample_001.jpg`
- mask: `sample_001.png`

Если маска не найдена, изображение все равно будет обработано, а метрики останутся пустыми.

## Запуск

### С ground truth

```bash
python -m src.run_experiment \
  --images_dir data/images \
  --masks_dir data/masks \
  --output_dir outputs/exp1 \
  --device cuda
```

### Без ground truth

```bash
python -m src.run_experiment \
  --images_dir data/images \
  --output_dir outputs/exp_no_gt \
  --device cuda
```

### С явным SAM checkpoint

```bash
python -m src.run_experiment \
  --images_dir data/images \
  --masks_dir data/masks \
  --output_dir outputs/exp1 \
  --device cuda \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b
```

## Обновление датасета in-place

Если нужно, чтобы новый пайплайн не просто сохранил результаты в `outputs/`, а обновил сам датасет, включи `--update_dataset`.

### Рекомендуемый безопасный режим для mixed-датасета

Для `dataset/abo_physics_natural_bg_v2` безопаснее всего писать обратно `main_object`, потому что датасет смешанный и не все объекты там ковры.

```bash
python -m src.run_experiment \
  --images_dir dataset/abo_physics_natural_bg_v2/images \
  --masks_dir dataset/abo_physics_natural_bg_v2/masks \
  --output_dir outputs/abo_physics_natural_bg_v2_grounded_sam \
  --device cuda \
  --update_dataset \
  --dataset_dir dataset/abo_physics_natural_bg_v2 \
  --dataset_update_method main_object \
  --write_dataset_previews
```

Что при этом обновится:

- `dataset/abo_physics_natural_bg_v2/masks/...`
- `dataset/abo_physics_natural_bg_v2/meta.json`
- `dataset/abo_physics_natural_bg_v2/meta.before_grounded_sam.json`
- `dataset/abo_physics_natural_bg_v2/grounded_sam_summary.json`
- `dataset/abo_physics_natural_bg_v2/masks_preview/...` при `--write_dataset_previews`

Важно:

- если ты указываешь `--masks_dir dataset/.../masks` одновременно с `--update_dataset`, то метрики в `results.csv` будут сравнением новых масок с предыдущими масками датасета, а не с внешним GT;
- `mask_path` и `mask_source` в `meta.json` будут переписаны на новый результат;
- предыдущие значения сохраняются в `seg_seed_mask_path` и `seg_seed_mask_source`.

### Другие режимы обновления датасета

Поддерживаются:

- `--dataset_update_method main_object`
- `--dataset_update_method carpet`
- `--dataset_update_method rug`
- `--dataset_update_method area_rug`
- `--dataset_update_method best_class`
- `--dataset_update_method best_available`

`best_class` выбирает лучший из `carpet / rug / area_rug` по сочетанию `dino_score`, `sam_score` и ненулевой маски.

## Основные CLI параметры

- `--images_dir`: папка с изображениями
- `--masks_dir`: папка с GT-масками
- `--output_dir`: папка с результатами
- `--device`: `auto`, `cpu`, `cuda`
- `--grounding_model_id`: Hugging Face model id или локальный путь для Grounding DINO
- `--sam_checkpoint`: путь к checkpoint SAM
- `--sam_model_type`: `vit_b`, `vit_l`, `vit_h`
- `--box_threshold`: threshold для боксов Grounding DINO
- `--text_threshold`: text threshold для Grounding DINO
- `--min_component_area`: фильтр маленьких компонент
- `--closing_kernel_size`: размер ядра closing
- `--closing_iterations`: число итераций closing
- `--update_dataset`: включить запись результата обратно в датасет
- `--dataset_dir`: корень датасета с `meta.json`
- `--dataset_update_method`: какой метод писать обратно в `mask_path`
- `--write_dataset_previews`: писать preview в `masks_preview`

## Структура выходов

```text
outputs/exp1/
  collages/
    ...
  logs/
    experiment.log
  masks/
    main_object/
    carpet/
    rug/
    area_rug/
  overlays/
    main_object/
    carpet/
    rug/
    area_rug/
  tables/
    results.csv
    summary_by_method.csv
    best_prompt_per_image.csv
    prompt_win_counts.csv
    main_object_vs_best_class.csv
    main_object_vs_best_class_summary.csv
```

## Что есть в `results.csv`

Колонки:

- `image_name`
- `method`
- `iou`
- `dice`
- `precision`
- `recall`
- `mask_area`
- `area_ratio`
- `dino_score`
- `sam_score`
- `inference_time`

## Аналитика

Проект дополнительно сохраняет:

- `best_prompt_per_image.csv`: какой из prompt-ов `carpet / rug / area rug` дал лучший IoU на каждом изображении;
- `prompt_win_counts.csv`: какой prompt чаще выигрывает;
- `main_object_vs_best_class.csv`: сравнение `main_object` против лучшего class-specific результата по каждому изображению;
- `main_object_vs_best_class_summary.csv`: сводка побед и средних метрик.

## Адаптеры API

Если у `Grounding DINO` или `SAM` изменится API, править нужно изолированные модули:

- `src/models/grounding_dino.py`
- `src/models/sam_wrapper.py`

Остальной пайплайн от этого не зависит.
