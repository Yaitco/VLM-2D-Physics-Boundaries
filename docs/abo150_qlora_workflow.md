# ABO150 QLoRA Workflow

## Что добавлено
Для `dataset/abo_150_expanded` теперь есть воспроизводимый путь:
- зафиксировать split `50 val / 100 train`;
- собрать per-property SFT dataset;
- дообучить QLoRA-адаптер;
- при желании сразу выгрузить адаптер на Hugging Face Hub;
- прогнать validation на тех же 50 объектах.

Готовый Colab-ноутбук под этот сценарий:
- [ABO150_Qwen3_QLoRA_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/notebooks/ABO150_Qwen3_QLoRA_Colab.ipynb)

Сейчас основной рекомендуемый базовый вариант для дообучения:
- `qwen3_vl_8b`

## Что нужно для Hugging Face Hub
Если хочешь, чтобы все адаптеры сразу уезжали на Hub, подготовь:
- `HF_TOKEN`
- опционально `HF_USERNAME`

В Colab их удобно положить в Secrets:
- `HF_TOKEN`
- `HF_USERNAME`

Если `HF_USERNAME` задан, скрипт сможет сам собирать repo id вида:
- `<HF_USERNAME>/<output_dir_name>`

## 1. Зафиксировать split
Скрипт:
[create_abo150_holdout_split.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/create_abo150_holdout_split.py)

Команда:
```bash
python scripts/create_abo150_holdout_split.py
```

По умолчанию создаётся:
- [val_ids.txt](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt)
- [train_ids.txt](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_150_expanded/splits/seed42_val50_train100/train_ids.txt)
- [manifest.json](/home/alexander/Projects/VLM-2D-Physics-Boundaries/dataset/abo_150_expanded/splits/seed42_val50_train100/manifest.json)

Этот split совместим с идеей `MAX_SAMPLES=50, RANDOM_SEED=42`, но теперь он зафиксирован явно.

## 2. Собрать train/val dataset для QLoRA
Скрипт:
[build_abo150_qlora_dataset.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/build_abo150_qlora_dataset.py)

Пример для одного свойства:
```bash
python scripts/build_abo150_qlora_dataset.py \
  --protocol-name full_expanded \
  --property-keys intrinsic.main_material \
  --train-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/train_ids.txt \
  --val-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --output-dir outputs/abo150_qlora_main_material
```

Пример для группы свойств:
```bash
python scripts/build_abo150_qlora_dataset.py \
  --protocol-name full_expanded \
  --property-keys intrinsic.main_material,intrinsic.transparency_class,affordance.breakable \
  --train-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/train_ids.txt \
  --val-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --output-dir outputs/abo150_qlora_material_transparency_breakable
```

На выходе будут:
- `train.jsonl`
- `val.jsonl`
- `manifest.json`

Формат данных:
- одна строка = один `(image, property)` training example
- prompt совпадает с inference prompt текущего per-property pipeline
- response — это target JSON

## 3. Обучить QLoRA
Скрипт:
[train_abo150_qlora.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/train_abo150_qlora.py)

### Локальный adapter без Hub
Пример:
```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/abo150_qlora_main_material/train.jsonl \
  --val-jsonl outputs/abo150_qlora_main_material/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/abo150_qlora_main_material/adapter \
  --num-train-epochs 6 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1
```

### Adapter сразу в Hugging Face Hub
Если `HF_USERNAME` уже задан:
```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/abo150_qlora_main_material/train.jsonl \
  --val-jsonl outputs/abo150_qlora_main_material/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/abo150_qwen3_main_material \
  --num-train-epochs 6 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --push-to-hub
```

Тогда repo id будет собран автоматически:
- `<HF_USERNAME>/abo150_qwen3_main_material`

Если хочешь задать repo id явно:
```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/abo150_qlora_main_material/train.jsonl \
  --val-jsonl outputs/abo150_qlora_main_material/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/abo150_qwen3_main_material \
  --push-to-hub \
  --hub-model-id your-hf-name/abo150-qwen3-main-material
```

На выходе будут:
- LoRA adapter weights
- `manifest.json`
- `runtime_model_config.json`
- `runtime_model_config.hub.json` при `--push-to-hub`

Файлы runtime-конфига нужны для unified validation pipeline:
- `runtime_model_config.json` — если adapter лежит локально
- `runtime_model_config.hub.json` — если adapter пушится на HF Hub

## 4. Валидировать adapter на тех же 50 объектах

### Через CLI
Для локального adapter:
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_150_expanded \
  --protocol-name full_expanded \
  --sample-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --model-config-path outputs/abo150_qlora_main_material/adapter/runtime_model_config.json \
  --custom-model-key qwen3_vl_8b_qlora_main_material \
  --variants raw
```

Для adapter из Hugging Face Hub:
```bash
python scripts/run_vlm_validation.py \
  --dataset-name abo_150_expanded \
  --protocol-name full_expanded \
  --sample-ids-path dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt \
  --model-config-path outputs/abo150_qwen3_main_material/runtime_model_config.hub.json \
  --custom-model-key qwen3_vl_8b_qlora_main_material_hub \
  --variants raw
```

### Через unified Colab notebook
В [Unified_VLM_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/notebooks/Unified_VLM_Validation_Colab.ipynb) теперь есть:
- `SAMPLE_IDS_PATH`
- `CUSTOM_MODEL_CONFIG_PATH`
- `CUSTOM_MODEL_KEY`

Для валидации adapter на 50 объектах:
```python
DATASET_NAME = 'abo_150_expanded'
PROTOCOL_NAME = 'full_expanded'

CUSTOM_MODEL_CONFIG_PATH = 'outputs/abo150_qwen3_main_material/runtime_model_config.hub.json'
CUSTOM_MODEL_KEY = 'qwen3_vl_8b_qlora_main_material_hub'
SELECTED_MODEL = CUSTOM_MODEL_KEY

SAMPLE_IDS_PATH = 'dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt'
PROPERTY_KEYS_MANIFEST_PATH = 'outputs/abo150_qwen3_main_material_dataset/manifest.json'
EVAL_VARIANTS = ['raw']
```

Если `PROPERTY_KEYS_MANIFEST_PATH` задан, unified validation notebook автоматически ограничит метрики только теми свойствами, на которых строился QLoRA trainset.

## 5. Валидировать больше базовых моделей на тех же 50 объектах
Используй:
```python
DATASET_NAME = 'abo_150_expanded'
PROTOCOL_NAME = 'full_expanded'
SAMPLE_IDS_PATH = 'dataset/abo_150_expanded/splits/seed42_val50_train100/val_ids.txt'
RUN_BATCH_GRID = True
```

И укажи `BATCH_MODEL_KEYS`.

## Практический совет
Для первого QLoRA-эксперимента лучше не брать весь `full_expanded` сразу.

Нормальный первый шаг:
- одно свойство:
  - `intrinsic.main_material`
или маленькая группа:
- `intrinsic.main_material`
- `intrinsic.transparency_class`
- `affordance.breakable`

Это даст понятный и быстрый пилот, на котором видно:
- даёт ли дообучение выигрыш вообще;
- улучшает ли оно `coverage`;
- не ломает ли JSON / формат ответа.

Если пилот покажет сигнал, следующий разумный шаг:
1. оставить базовую модель `qwen3_vl_8b`;
2. расширить trainset до 2–3 свойств;
3. все новые adapter-версии сразу пушить на HF Hub, чтобы не путаться в локальных папках.

Если нужен не встроенный ABO150-сценарий, а свои данные и свои свойства, см.:
- [custom_datasets_and_properties.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/custom_datasets_and_properties.md)
