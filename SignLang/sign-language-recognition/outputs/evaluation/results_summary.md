# Classification results

Selected model: **LogisticRegression**.

Selection used validation macro F1. This same model was fitted on training rows only.

Test images: **504**. Classes: **36**.

| Metric | Test value |
| --- | ---: |
| accuracy | 0.7758 |
| macro_precision | 0.7908 |
| macro_recall | 0.7837 |
| macro_f1 | 0.7672 |
| weighted_precision | 0.8057 |
| weighted_recall | 0.7758 |
| weighted_f1 | 0.7715 |

## Most frequent directional confusions

| True label | Predicted label | Count |
| --- | --- | ---: |
| W | 6 | 9 |
| 2 | V | 6 |
| O | C | 6 |
| X | Z | 6 |
| L | D | 4 |
| O | 0 | 4 |
| R | U | 4 |
| S | M | 4 |
| Z | 1 | 4 |
| J | G | 3 |

## Per-class interpretation

Highest observed F1: 3 (1.000, n=10), Y (1.000, n=18), E (1.000, n=16), 8 (1.000, n=9), F (0.973, n=18)

Lowest observed F1: Z (0.320, n=12), T (0.400, n=15), O (0.462, n=18), 2 (0.471, n=10), 6 (0.500, n=6)

Small per-class test counts make these estimates variable.

## Interpretation limits

Extraction coverage and classification accuracy measure different stages. Do not multiply them to claim end-to-end accuracy.
Webcam performance must be checked using new captures. Display smoothing and rejection thresholds are not part of these metrics.
Test results describe this selected model. Further changes based on these errors require a new untouched evaluation set.

- Photo filename families are source proxies; verified signer IDs are not available for all images.
- No overlap of the defined groups does not prove complete signer independence.
- Image hashes detect exact decoded-image duplicates, not every near-duplicate.
- The split covers accepted single-hand features; report extraction coverage separately from classifier accuracy.
- Exploratory preprocessing used dataset samples before this split; a new independent capture is needed for an untouched external evaluation.
