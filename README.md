# VLM-2D-Physics-Boundaries

Текущая структура репозитория упрощена вокруг одного основного сценария: `per_property` валидация VLM на `ABO150`.

## Навигация
- Основной Colab-ноутбук: [ABO150_Validation_Colab.ipynb](/home/alexander/Projects/VLM-2D-Physics-Boundaries/ABO150_Validation_Colab.ipynb)
- Validation pipeline: [docs/validation_pipeline.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/validation_pipeline.md)
- Colab runbook: [docs/colab_runbook.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/colab_runbook.md)
- ABO natural-background subset: [docs/abo_natural_background_subset.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/abo_natural_background_subset.md)
- SAM mask generation: [docs/sam_mask_generation.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/docs/sam_mask_generation.md)
- План на 5 дней: [coursework_5day_plan.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/coursework_5day_plan.md)
- Отчёт по Дню 1: [day1_requirements.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day1_requirements.md), [day1_code_audit.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day1_code_audit.md), [day1_experiment_matrix.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day1_experiment_matrix.md)
- Отчёт по Дню 2: [day2_report.md](/home/alexander/Projects/VLM-2D-Physics-Boundaries/day2_report.md)


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
  --progress-every-records 5000 \
  --progress-every-selected 25
