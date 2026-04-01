# Свои Датасеты И Свои Свойства

Этот документ нужен для двух сценариев:
- ты хочешь прогонять zero-shot / few-shot инференс на своём датасете;
- ты хочешь дообучать QLoRA-адаптеры под свои свойства и свои данные.

Ниже описан самый прямой путь с минимальными изменениями в коде.

## 1. Что уже умеет проект

Сейчас в проекте есть две почти независимые части:
- **inference / validation pipeline** — `scripts/run_vlm_validation.py` + `vlm_pipeline/*`;
- **QLoRA trainer** — `scripts/train_abo150_qlora.py`.

Важно:
- trainer уже умеет обучаться на любом JSONL нужного формата;
- а вот loader и CLI для инференса пока знают только встроенные datasets/protocols.

То есть для кастомных данных нужно:
1. описать датасет;
2. описать схему свойств;
3. либо подключить их к pipeline, либо экспортировать готовый SFT JSONL для обучения.

## 2. Минимальные требования

### Для инференса
- Python `3.10+`
- локальные изображения
- `meta.json` с GT по свойствам
- YAML со схемой свойств
- GPU крайне желателен

### Для QLoRA
- всё выше;
- плюс train/val split;
- плюс GPU с достаточной памятью;
- при выгрузке адаптера на Hub: `HF_TOKEN`, опционально `HF_USERNAME`.

## 3. Рекомендуемый формат кастомного датасета

Самый простой путь — использовать тот же тип данных, что у `abo_physics_natural_bg_v2`, то есть `dataset_type = "meta_json"`.

### Рекомендуемая структура
```text
dataset/
  my_dataset/
    meta.json
    images/
      0001.jpg
      0002.jpg
    masks/
      0001.png
```

### Минимальный формат `meta.json`
`meta.json` должен быть списком объектов:

```json
[
  {
    "image_id": "0001",
    "path": "my_dataset/images/0001.jpg",
    "notes": "optional caption or notes",
    "primary_object": "chair",
    "properties": {
      "material": "wood",
      "rigidity": "rigid",
      "transparency": "opaque"
    },
    "mask_path": "my_dataset/masks/0001.png",
    "seg_preview_path": "my_dataset/masks_preview/0001.jpg"
  }
]
```

### Важное замечание про пути
Для `meta_json` loader резолвит относительные пути относительно папки `dataset/`, а не относительно `dataset/my_dataset/`.

То есть корректно так:
- `my_dataset/images/0001.jpg`
- `my_dataset/masks/0001.png`

А не так:
- `images/0001.jpg`

## 4. Как описать свои свойства

Для компактных property schemes проще всего использовать YAML такого вида:

```yaml
properties:
  material:
    type: categorical
    enum: [wood, metal, plastic, fabric]
    description: Main material of the object.

  rigidity:
    type: categorical
    enum: [rigid, flexible, soft]
    description: Deformation behavior under light force.

  transparency:
    type: categorical
    enum: [opaque, transparent]
    description: Whether the object transmits light.
```

Положи файл, например, в:
- `configs/my_dataset_properties.yaml`

### Если имя в `meta.json` и имя в prompt различаются
Используй `source_name`:

```yaml
properties:
  main_material:
    type: categorical
    source_name: material
    enum: [wood, metal, plastic, fabric]
    description: Main material of the object.
```

Тогда:
- ключ свойства в pipeline будет `main_material`;
- GT будет браться из `properties.material`.

## 5. Как подключить новый датасет к inference pipeline

### Шаг 1. Добавить DatasetContext
Файл:
- [vlm_pipeline/registry.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/registry.py)

Добавь ветку в `get_dataset_context(...)`:

```python
if dataset_name == "my_dataset":
    return DatasetContext(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        dataset_type="meta_json",
        meta_path=dataset_dir / "meta.json",
        reports_dir=ROOT_DIR / "reports_my_dataset",
    )
```

### Шаг 2. Разрешить имя датасета в CLI
Файл:
- [scripts/run_vlm_validation.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/run_vlm_validation.py)

В аргументе `--dataset-name` добавь `my_dataset` в `choices`.

### Шаг 3. Подключить новый protocol
Файл:
- [vlm_pipeline/specs.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/vlm_pipeline/specs.py)

Добавь константу:

```python
MY_DATASET_SCHEMA_PATH = ROOT_DIR / 'configs' / 'my_dataset_properties.yaml'
```

И ветку в `load_protocol_property_specs(...)`:

```python
if protocol_name == 'my_dataset':
    return load_compact_property_specs(MY_DATASET_SCHEMA_PATH)
```

### Шаг 4. Запуск
```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-key qwen3_vl_8b \
  --variants raw \
  --max-samples 100 \
  --random-seed 42
```

## 6. Как ограничить запуск только своими свойствами

Если свойства уже есть в протоколе, но нужен только поднабор:

```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-key qwen3_vl_8b \
  --variants raw \
  --property-keys material,rigidity,transparency
```

Либо можно передать `manifest.json`, в котором есть поле `property_keys`:

```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-key qwen3_vl_8b \
  --property-keys-manifest-path outputs/my_manifest.json \
  --variants raw
```

## 7. Как подготовить свои данные для QLoRA

Есть два пути.

### Вариант A. Адаптировать dataset builder
Если твой датасет уже подключён к pipeline, можно сделать отдельную копию логики из:
- [scripts/build_abo150_qlora_dataset.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/build_abo150_qlora_dataset.py)

и заменить там:
- `dataset-name`
- `protocol-name`
- источник `sample_ids_path`

Это удобно, если ты хочешь собирать SFT JSONL автоматически из `meta.json` и YAML-схемы.

### Вариант B. Сразу экспортировать JSONL вручную
`train_abo150_qlora.py` ждёт JSONL с такими полями:
- `image_id`
- `image_path`
- `property_key`
- `prompt`
- `response`
- `gt_value`

Минимальный пример строки:

```json
{
  "image_id": "0001",
  "image_path": "/absolute/path/to/dataset/my_dataset/images/0001.jpg",
  "property_key": "material",
  "prompt": "You are given an image_id and a property to predict...",
  "response": "{\"image_id\": \"0001\", \"properties\": {\"material\": \"wood\"}}",
  "gt_value": "wood"
}
```

Этот вариант удобен, если:
- у тебя уже есть собственный data preprocessing;
- не хочется патчить loader репозитория;
- нужно быстро дообучить адаптер на внешнем датасете.

## 8. Как строить prompt и response без ручного копирования формата

Если ты собираешь JSONL внутри репозитория, не генерируй prompt вручную. Лучше использовать:
- `build_property_prompt(...)`
- `build_demo_response_payload(...)`

Они уже используются в:
- [scripts/build_abo150_qlora_dataset.py](/home/alexander/Projects/VLM-2D-Physics-Boundaries/scripts/build_abo150_qlora_dataset.py)

Это снижает риск, что train-format и inference-format разойдутся.

## 9. Как обучить адаптер на своих данных

Когда у тебя уже есть `train.jsonl` и `val.jsonl`, дальше pipeline обычный:

```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/my_dataset_qlora/train.jsonl \
  --val-jsonl outputs/my_dataset_qlora/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/my_dataset_qwen3_adapter \
  --num-train-epochs 6 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1
```

### Если хочешь сразу выгрузить адаптер на Hub
```bash
python scripts/train_abo150_qlora.py \
  --train-jsonl outputs/my_dataset_qlora/train.jsonl \
  --val-jsonl outputs/my_dataset_qlora/val.jsonl \
  --model-key qwen3_vl_8b \
  --output-dir outputs/my_dataset_qwen3_adapter \
  --push-to-hub \
  --hub-model-id your-hf-name/my-dataset-qwen3-adapter
```

## 10. Как провалидировать свой адаптер

Если датасет уже подключён к registry и protocol подключён к specs:

```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-config-path outputs/my_dataset_qwen3_adapter/runtime_model_config.json \
  --custom-model-key qwen3_vl_8b_my_dataset_adapter \
  --variants raw
```

Если адаптер лежит на Hugging Face Hub:

```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-config-path outputs/my_dataset_qwen3_adapter/runtime_model_config.hub.json \
  --custom-model-key qwen3_vl_8b_my_dataset_adapter_hub \
  --variants raw
```

Если хочешь считать метрики только по тем свойствам, на которых обучался адаптер:

```bash
python scripts/run_vlm_validation.py \
  --dataset-name my_dataset \
  --protocol-name my_dataset \
  --model-config-path outputs/my_dataset_qwen3_adapter/runtime_model_config.json \
  --custom-model-key qwen3_vl_8b_my_dataset_adapter \
  --property-keys-manifest-path outputs/my_dataset_qlora/manifest.json \
  --variants raw
```

## 11. Практические советы

### Для инференса
- сначала запусти `MAX_SAMPLES = 5` или короткий subset;
- сначала используй `raw`, а маски добавляй уже после sanity-check;
- если свойств много, начинай с поднабора через `--property-keys`.

### Для QLoRA
- сначала не бери весь ontology space;
- начни с 1–5 свойств, которые визуально наблюдаемы и не слишком дисбалансны;
- держи train-format максимально близким к inference-format.

### Для transfer
Если адаптер обучен на одном датасете, а валидируется на другом, не сравнивай его со средним по всем свойствам. Сравнение должно идти:
- либо по тем же property keys;
- либо по тому же collapsed target space.

## 12. Ограничения текущей реализации

- `run_vlm_validation.py` пока не полностью универсален через CLI: новый датасет и новый protocol нужно один раз зарегистрировать в коде.
- `build_abo150_qlora_dataset.py` по имени и дефолтным настройкам заточен под ABO150.
- `train_abo150_qlora.py --help` сейчас может падать раньше `argparse`, если окружение не готово к импортам `transformers`/`Trainer`; для обучения это не критично, но это стоит иметь в виду.

## 13. Если хочется минимальных патчей

Самый дешёвый рабочий путь такой:
1. завести `meta.json`-датасет;
2. добавить одну ветку в `registry.py`;
3. добавить один YAML и одну ветку в `specs.py`;
4. для обучения генерировать SFT JSONL напрямую.

Этого уже хватает, чтобы использовать текущий репозиторий как каркас под свой benchmark и под свои property adapters.
