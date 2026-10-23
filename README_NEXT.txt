NSCLC GROUPED REFINEMENT — NEXT STAGE
=====================================

Purpose
-------
This is a separate validation-controlled refinement experiment. It preserves
all five original direct grouped checkpoints. It focuses on the two weaknesses
confirmed independently on validation: Macrophages and Necrosis. It also adds
hard-pixel and boundary supervision and penalizes Stroma false positives.

Do not claim 85% before a valid evaluation reports it. Do not repeatedly tune
against the untouched holdout. The test labels are not used anywhere here.

Run order
---------
1. Double-click RUN_REFINEMENT_5FOLD.bat.
2. After all five folds finish, double-click CHECK_REFINEMENT_VALIDATION.bat.
3. Upload Downloads\REFINEMENT_VALIDATION_DIAGNOSTICS.zip to ChatGPT.
4. Do not run another holdout evaluation until the validation gate is checked.

Expected time
-------------
Approximately 1.5 to 3 hours per fold on the RTX 4050, often less with early
stopping. The laptop may stay offline because all required weights are cached.
Keep it plugged in and prevent sleep.

Scientific rule fixed before retesting
--------------------------------------
A new holdout evaluation is allowed only if mean validation macro F1 improves
by at least 0.005, mean accuracy drops no more than 0.003, and the weaker of
Macrophage/Necrosis validation F1 improves. This protects the untouched test.
