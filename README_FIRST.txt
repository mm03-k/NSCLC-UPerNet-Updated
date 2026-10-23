NSCLC DIRECT FIVE-CLASS GROUPED UPerNet — 85% TARGET
===================================================

Purpose
-------
This is a new, separate experiment designed to improve BOTH:
  1) five-class pixel accuracy, and
  2) five-class macro F1/Dice.

The target is 85% for both metrics. It is a target, not a guaranteed or
pre-written result. The scripts report the exact untouched-holdout result.

Prespecified five tissue groups
-------------------------------
1. Tumor
2. Stroma + inflammation
3. Macrophages
4. Necrosis
5. Rest

Scientific safeguards
---------------------
- The official 139-ROI test set is untouched during training and selection.
- Each fold is selected on its official validation split.
- The selection score is the lower of validation accuracy and validation
  macro F1, so one metric cannot hide weakness in the other.
- Existing 16-class checkpoints are only used as initialization.
- New checkpoints use separate output folders; prior results are preserved.

After extracting this ZIP into NSCLC_UPerNet
--------------------------------------------
1. Double-click RUN_GROUPED_5FOLD.bat.
2. Let all five folds finish. On the RTX 4050 this can take several hours.
3. Double-click RUN_GROUPED_EVALUATION.bat once all five folds exist.
4. Evaluation can take roughly 1-2 hours because it uses 50% overlap,
   five models, and eight test-time transforms.

Final files
-----------
outputs\upernet_swin_tiny_grouped_5fold_ensemble\test_results_overlap50_d4

The important file is metrics.json. The command window also prints:
  Pixel accuracy
  Macro F1/Dice
  Weighted F1
  Mean IoU
  85% BOTH target: PASSED or NOT YET PASSED

Do not edit metrics.json and do not repeatedly tune on the 139 test masks.
If the result is below target, share the genuine metrics and tune only using
the validation-fold histories before performing one final locked test run.
