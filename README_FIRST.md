# NSCLC UPerNet — IGNITE v2

This code implements multi-class semantic tissue segmentation on the official
IGNITE H&E split using UPerNet with a Swin-Tiny backbone.

## Data-safety rules built into the code

- `data_overview.csv` is the source of truth, not the raw folder count.
- Only rows with `task == he_tissue_segmentation` are used.
- `split == test` is never loaded by the training program.
- `fold0` is used for validation by default; the other four official folds are
  used for training.
- Raw labels `0` (`Unannotated`) and `255` (`ignore`) are excluded. Raw labels `1..16` are remapped to
  model labels `0..15`.
- BatchNorm running statistics are frozen during batch-size-one fine-tuning,
  while all other UPerNet parameters remain trainable.
- Image and mask paths are taken directly from the official metadata, preventing
  normal/context copies from being treated as independent samples.
- During final holdout evaluation, the corresponding `_with_context` variant is
  used when present; its unannotated border is excluded from the metrics.

## Commands (run from the NSCLC_UPerNet folder)

Activate the existing environment first:

```bat
.venv\Scripts\activate
```

Install only the remaining Python packages (this does not replace CUDA PyTorch):

```bat
python -m pip install -r requirements.txt
```

Validate the dataset and generate class weights:

```bat
python src\01_validate_dataset.py
```

Run one forward/backward smoke test:

```bat
python src\02_smoke_test.py
```

The smoke test downloads `openmmlab/upernet-swin-tiny` once (roughly a few
hundred MB). It is cached afterwards.

Train:

```bat
python src\03_train.py
```

Resume an interrupted run:

```bat
python src\03_train.py --resume outputs\upernet_swin_tiny_hybrid_fold0\last.pt
```

Evaluate once on the untouched holdout test set:

```bat
python src\04_evaluate.py
```

Results are written to:

```text
outputs/upernet_swin_tiny_hybrid_fold0/
```

Important outputs include `best.pt`, `training_history.csv`,
`training_curves.png`, `test_results/metrics.json`, `per_class_metrics.csv`,
`confusion_matrix.png`, `per_class_f1.png`, and qualitative predictions.
The final metrics also include the paper-style TIL grouping (tumor,
stroma+inflammation, macrophages, necrosis, and rest).

## If the RTX 4050 reports out-of-memory

Close games and browsers using the GPU. If it still fails, change
`"patch_size": 512` to `"patch_size": 384` in `config.json`, rerun the smoke
test, and train with that setting. Do not silently change the size after results
are generated; record the final value in the paper.
