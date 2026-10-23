# NSCLC UPerNet — Updated research source code

Research code for **Multi-Class Tissue Segmentation in Non-Small Cell Lung Cancer Histopathology Images Using UPerNet** (IGNITE H&E).

This archive combines the available base Swin-Tiny UPerNet pipeline with two later experimental code updates:

1. **Original 16-class segmentation** — `src/01_validate_dataset.py` through `src/04_evaluate.py`, `config.json`.
2. **Five-group 85% *target* experiment** — `src/12_train_grouped.py`, `src/grouped_dataset.py`, `src/13_evaluate_grouped_ensemble.py`, configs `config_grouped_fold0.json` ... `config_grouped_fold4.json`, and the `RUN_GROUPED_*.bat` scripts. Five tissue groups: tumor, stroma/inflammation, macrophages, necrosis, rest.
3. **Refinement v2** — `src/14_refine_grouped.py`, `src/15_check_refinement.py`, `config_refined_fold0.json` ... `config_refined_fold4.json`, and `RUN_REFINEMENT_5FOLD.bat` / `CHECK_REFINEMENT_VALIDATION.bat`. Refinement focuses on macrophage and necrosis segmentation, boundary and hard-pixel supervision.

## Important status

**85% is an experimental goal, not a verified result.** No trained weights, model outputs, evaluation metrics or dataset are in this source archive. Do not describe the target as attained without a measured `metrics.json` from a genuine evaluation. The last documented 16-class selective-hybrid holdout results were **80.64% pixel accuracy** and **82.09% grouped F1**; these do *not* verify 85% for the newer grouped-model experiments.

## Prerequisites

- Windows machine with compatible GPU/CUDA and PyTorch installation
- Python and the libraries listed in `requirements.txt`
- The official IGNITE H&E data, arranged under `data/raw/` according to `config.json`
- Trained 16-class fold checkpoints are referenced in the `config_grouped_fold*.json` files and are **not supplied** here.
- The base archive contains one original 16-class fold config (`config.json`), not all original 16-class fold configs. If you need to train all five 16-class folds from scratch, obtain the remaining configurations from the original research working folder or prior repository; do not assume these have been trained.

Run order and dataset-safety notes are in `README_FIRST.md`, `README_FIRST.txt`, and `README_NEXT.txt`. The later update is validation-controlled: do not repeatedly tune against the locked 139-ROI holdout.

## Upload on a different GitHub account

1. Sign in to the **GitHub account you want the professor to see**.
2. Create a new public repository, e.g. `NSCLC-UPerNet-Updated` (without generating a README).
3. Extract this ZIP. On the new empty repo, select **uploading an existing file** and drag the **contents** of the `NSCLC_UPerNet_Updated` folder (not the outer ZIP). Preserve `src/` structure and config files.
4. Commit the files; send your professor the new repository URL, e.g. `https://github.com/<NEW-USERNAME>/NSCLC-UPerNet-Updated`.
5. If GitHub’s browser uploader cannot upload all folders correctly, use GitHub Desktop to publish the extracted folder under your other account.

Never upload patient-identifiable data, private credentials, entire datasets, checkpoints or unreviewed outputs. Confirm the dataset license and repository permissions before publishing.
