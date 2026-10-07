# Verification of the continuation package

Date: 2026-09-16.

This record describes the assistant-side run on Ziad's uploaded CSV files. **Ziad's Windows test is still pending, starting with step 07.** Webcam hardware and real MediaPipe inference were not exercised in this environment.

## Real input data and split

- Original audit: 3607 images.
- Accepted padded feature table: 3402 rows, 43 columns.
- Excluded pending hand-selection review: 3 rows.
- Removed extra exact-image duplicate: 1 row.
- Clean data: 3398 rows.
- Train: 2387; validation: 507; test: 504.
- All splits contain all 36 labels.
- No overlap of the 12 defined filename-based source groups or exact image hashes between splits.
- Each generated row was reconciled with its original feature row and label.
- Filename groups are source proxies for part of the collection; full signer independence is not established.

## Model selection, before test evaluation

| Model | Training accuracy | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: |
| Decision Tree | 0.924173 | 0.751479 | 0.727771 |
| Random Forest | 1.000000 | 0.850099 | 0.830965 |
| Logistic Regression | 0.956012 | 0.907298 | 0.883543 |

The fixed rule selected **Logistic Regression** by validation macro F1. Its StandardScaler saw exactly 2387 training rows. The saved model remains fitted on training data only. No classifier settings were changed after inspecting the test results.

## Test evaluation of the selected model

| Metric | Value |
| --- | ---: |
| Correct predictions | 391 / 504 |
| Accuracy | 0.775794 |
| Macro precision | 0.7919 |
| Macro recall | 0.7837 |
| Macro F1 | 0.7670 |
| Weighted precision | 0.8071 |
| Weighted recall | 0.7758 |
| Weighted F1 | 0.7712 |

The exact values and predictions-derived reports are in `reference_run`. Confusion-matrix totals and diagonal counts were independently reconciled with the prediction table.

Examples of directional errors: W to 6 occurred 9 times; 2 to V occurred 6 times; O to C occurred 6 times; X to Z occurred 6 times. These are observations, not confirmed explanations of their causes.

These metrics evaluate the classifier on accepted, cleaned static-image features. They are not webcam accuracy or complete end-to-end system accuracy. Preprocessing was explored on dataset samples before fixing the split, so a new independent capture is needed for a fully untouched external evaluation.

## Additional checks

- All new Python files parsed successfully.
- Saved model reloaded and produced valid 36-class probability vectors in the correct feature order.
- Inference data flow was checked with controlled detector outputs: BGR-to-RGB order, padded input dimensions, 42-feature normalization, inverse mapping for drawing, and zero/one/two-hand handling.
- Degenerate palm geometry was rejected.
- Display smoothing warms up correctly and clears previous predictions when hand detection is absent or invalid.
- The sequential runner stops at a failed stage; later stages are not executed.
- Model-comparison and confusion-matrix plots were visually inspected.

The controlled inference checks do **not** constitute a real MediaPipe image test or camera test. OpenCV and MediaPipe are installed in Ziad's environment, but were not available in this verification environment. The remaining local test is documented in `START_HERE_AR.md`.

## Verification environment

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| scikit-learn | 1.8.0 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| joblib | 1.5.3 |

Ziad's Python is 3.12.10. Training in his existing virtual environment avoids distributing a pickled estimator across potentially different package versions. Compare the resulting reports after the Windows run; minor numerical differences may occur.
