# VLM-2D-Physics-Boundaries

Репозиторий курсовой работы по оценке vision-language моделей на задаче извлечения физических свойств объектов из 2D-изображений.

В проекте собраны:
- единый validation pipeline для `per-property` инференса;
- zero-shot и few-shot сравнение нескольких VLM;
- абляции по вариантам визуального входа: `raw`, `mask_overlay`, `masked`;
- QLoRA-дообучение адаптеров под выбранные группы свойств;
- служебные скрипты для подготовки subset'ов, масок и ручного review.

Основная активная логика живёт в `vlm_pipeline/` и `scripts/`. Папка `src/` содержит старые и вспомогательные сегментационные наработки; это уже не главный entrypoint репозитория.

## Что именно делает проект

Мы рассматриваем задачу так:
- есть изображение объекта;
- есть список физических свойств и допустимых значений;
- модель получает короткий prompt на одно свойство за раз;
- ответ нормализуется к фиксированному enum/boolean space;
- дальше считаются `accuracy`, `macro-F1`, `coverage` и вспомогательные метрики.

Такой режим нужен, чтобы:
- сравнивать разные модели на одной и той же постановке;
- отдельно смотреть, где модель не знает ответ, а где отвечает неверно;
- запускать узкие QLoRA-адаптеры под конкретные группы свойств.

## Поддерживаемые датасеты и протоколы

### Датасеты
- `abo_150_expanded`
  Небольшой, но богато размеченный subset ABO с широкой ontology свойств. Используется для zero-shot на расширенной схеме и для QLoRA.
- `abo_physics_natural_bg_v2`
  Natural-background subset с компактным набором из пяти свойств. Используется как основной benchmark для сравнения моделей и transfer-проверок.

### Протоколы свойств
- `narrow_core`
  Узкий набор свойств для быстрых baseline-сравнений.
- `full_expanded`
  Полная ontology `abo_150_expanded`.
- `pdf_compact`
  Компактный coursework-протокол, полученный через mapping из ABO ontology.
- `natural_bg_v2`
  Компактный 5-property протокол для `abo_physics_natural_bg_v2`.
- `natural_bg_v2_main_material`
  Узкий протокол только для `main_material`-transfer.
- `abo150_natural_bg_v2_transfer`
  ABO150-протокол, схлопнутый в тот же label-space, что и `natural_bg_v2`.

## Поддерживаемые модели

Из коробки в registry есть:
- `qwen3_vl_8b`
- `qwen2_5_vl_7b`
- `qwen2_5_vl_3b`
- `qwen2_vl_2b`
- `llava_onevision_1_5_8b`

Все они подключаются через единый runtime layer. Для дообученных моделей можно добавлять кастомный runtime config через `--model-config-path`.

## Варианты визуального входа

- `raw` — исходное изображение;
- `mask_overlay` — изображение с наложением маски;
- `masked` — объект, вырезанный по маске на однотонный фон.

Варианты `mask_overlay` и `masked` доступны только если в sample есть `mask_path`.

## Структура репозитория

```text
configs/                 схемы свойств и manifests
dataset/                 локальные датасеты и subsets
docs/                    инструкции по пайплайну и воспроизведению
notebooks/               Colab-ноутбуки
outputs/                 train/eval outputs, adapters, manifests
scripts/                 CLI entrypoints
vlm_pipeline/            активное ядро validation pipeline
Coursework.pdf           собранный PDF курсовой
```

Ключевые файлы:
- [scripts/run_vlm_validation.py](scripts/run_vlm_validation.py)
- [scripts/build_abo150_qlora_dataset.py](scripts/build_abo150_qlora_dataset.py)
- [scripts/train_abo150_qlora.py](scripts/train_abo150_qlora.py)
- [vlm_pipeline/registry.py](vlm_pipeline/registry.py)
- [vlm_pipeline/specs.py](vlm_pipeline/specs.py)
- [vlm_pipeline/datasets.py](vlm_pipeline/datasets.py)

## Требования и реквизиты

### Окружение
Минимально:
- Python `3.10+`;
- Linux или Colab;
- GPU для практического инференса и особенно для QLoRA.

Рекомендации по ресурсам:
- zero-shot инференс 7B/8B моделей удобнее делать на GPU с 12–16 GB VRAM и выше;
- QLoRA-дообучение стабильнее запускать на 16–24 GB VRAM и выше;
- для долгих запусков и выгрузки адаптеров на Hub удобен Google Colab.

### Установка
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Внешние токены и секреты
Обязательны не всегда, но на практике полезны:
- `HF_TOKEN` — если нужны загрузка gated-моделей или push адаптеров на Hugging Face Hub;
- `HF_USERNAME` — чтобы автоматически собирать `repo_id` для адаптеров;
- `comet_api_key`, `comet_workspace` — если хочешь логирование в Comet.

### Что должно лежать локально
Проект ожидает локальные датасеты в `dataset/`. Для встроенных сценариев важны:
- `dataset/abo_150_expanded`
- `dataset/abo_physics_natural_bg_v2`

## Быстрый старт: zero-shot валидация

### 1. Запуск на `abo_physics_natural_bg_v2`
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_physics_natural_bg_v2 \
  --protocol-name natural_bg_v2 \
  --model-key qwen3_vl_8b \
  --variants raw \
  --max-samples 100 \
  --random-seed 42
```

### 2. Запуск на вручную approved subset
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_physics_natural_bg_v2 \
  --protocol-name natural_bg_v2 \
  --model-key qwen3_vl_8b \
  --variants raw \
  --meta-override-path dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_final_kept_meta.json
```

### 3. Запуск на `abo_150_expanded`
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_150_expanded \
  --protocol-name narrow_core \
  --model-key qwen3_vl_8b \
  --variants raw \
  --max-samples 50 \
  --random-seed 42
```

### Что сохраняется
Для каждого `model / variant` сохраняются:
- `per_sample_predictions.csv`
- `property_metrics.csv`
- `summary.json`

## Быстрый старт: QLoRA на встроенном ABO150

### 1. Зафиксировать split
```bash
python scripts/create_abo150_holdout_split.py
```

### 2. Построить train/val dataset под выбранные свойства
```bash
python scripts/build_abo150_qlora_dataset.py \
  --protocol-name full_expanded \
  --property-keys intrinsic.main_material,intrinsic.transparency_class \
  --train-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/train_ids.txt \
  --val-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --output-dir outputs/abo150_qwen3_material_transparency_dataset
```

### 3. Обучить адаптер
```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/abo150_qwen3_material_transparency_dataset/train.jsonl \
  --val-jsonl outputs/abo150_qwen3_material_transparency_dataset/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/abo150_qwen3_material_transparency \
  --num-train-epochs 6 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1
```

### 4. Провалидировать адаптер
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_150_expanded \
  --protocol-name full_expanded \
  --sample-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --model-config-path outputs/abo150_qwen3_material_transparency/runtime_model_config.json \
  --custom-model-key qwen3_vl_8b_qlora_material_transparency \
  --property-keys-manifest-path outputs/abo150_qwen3_material_transparency_dataset/manifest.json \
  --variants raw
```

## Дообучение и инференс на своих датасетах и своих свойствах

Это поддерживается, но сейчас есть важная оговорка:
- `train_abo150_qlora.py` уже достаточно общий и принимает любой SFT JSONL в нужном формате;
- а вот `build_abo150_qlora_dataset.py` и `run_vlm_validation.py` по умолчанию знают только про встроенные датасеты.

Нормальный путь для кастомного проекта такой:
1. подготовить свой датасет в `meta.json`-формате;
2. добавить новый `DatasetContext` в [vlm_pipeline/registry.py](vlm_pipeline/registry.py);
3. описать свои свойства в YAML-конфиге и зарегистрировать новый protocol в [vlm_pipeline/specs.py](vlm_pipeline/specs.py);
4. для инференса использовать `run_vlm_validation.py`;
5. для обучения либо расширить dataset builder, либо сразу экспортировать `train.jsonl` / `val.jsonl` в формате, который ждёт [train_abo150_qlora.py](scripts/train_abo150_qlora.py).

Подробная инструкция с шаблонами лежит здесь:
- [docs/custom_datasets_and_properties.md](docs/custom_datasets_and_properties.md)

## Colab

Готовые ноутбуки:
- [notebooks/Unified_VLM_Validation_Colab.ipynb](notebooks/Unified_VLM_Validation_Colab.ipynb)
- [notebooks/ABO150_Qwen3_QLoRA_Colab.ipynb](notebooks/ABO150_Qwen3_QLoRA_Colab.ipynb)

Быстрый runbook:
- [docs/colab_runbook.md](docs/colab_runbook.md)

## Полезные документы

- [docs/validation_pipeline.md](docs/validation_pipeline.md) — как устроен unified validation pipeline;
- [docs/abo150_qlora_workflow.md](docs/abo150_qlora_workflow.md) — воспроизводимый QLoRA workflow на ABO150;
- [docs/custom_datasets_and_properties.md](docs/custom_datasets_and_properties.md) — как завести свои данные и свои свойства;
- [docs/manual_segmentation_review.md](docs/manual_segmentation_review.md) — ручной review масок;
- [docs/sam_mask_generation.md](docs/sam_mask_generation.md) — генерация SAM-масок;
- [docs/abo_natural_background_subset.md](docs/abo_natural_background_subset.md) — подготовка natural background subset.

## Ограничения текущей версии

- `run_vlm_validation.py` по умолчанию ограничен встроенными dataset names в CLI; для нового датасета нужен небольшой registry/config patch.
- `build_abo150_qlora_dataset.py` ориентирован на `abo_150_expanded`; для кастомного датасета нужно либо адаптировать builder, либо готовить JSONL напрямую.
- Несколько документов и скриптов вокруг сегментации сохранились как вспомогательные артефакты исследования; основной evaluation loop находится в `vlm_pipeline/`.

## Курсовая

- Текст работы: [Coursework.pdf](Coursework.pdf)

Если хочется быстро понять, с чего начинать в коде, то лучший маршрут такой:
1. [scripts/run_vlm_validation.py](scripts/run_vlm_validation.py)
2. [vlm_pipeline/evaluation.py](vlm_pipeline/evaluation.py)
3. [vlm_pipeline/datasets.py](vlm_pipeline/datasets.py)
4. [scripts/train_abo150_qlora.py](scripts/train_abo150_qlora.py)
