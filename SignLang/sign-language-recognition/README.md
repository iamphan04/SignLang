# Real-Time Sign Language Recognition

A real-time static hand gesture recognition system built with Python, MediaPipe, OpenCV, and classical machine learning.

The system recognizes 36 static labels:

- Digits: `0-9`
- Letters: `A-Z`

## Project Overview

Each image or webcam frame passes through the following pipeline:

1. Capture or load an image.
2. Add black padding to improve detection of tightly cropped hands.
3. Detect one hand using MediaPipe Hand Landmarker.
4. Extract 21 hand landmarks.
5. Use the X and Y coordinates to produce 42 features.
6. Normalize landmark position relative to the wrist.
7. Normalize scale using the wrist-to-middle-finger-base distance.
8. Predict the gesture using a trained classical machine-learning classifier.

## Dataset

- Original images: 3,607
- Classes: 36
- Accepted after padded extraction: 3,402
- Clean rows after review and duplicate removal: 3,398
- Training rows: 2,387
- Validation rows: 507
- Test rows: 504

The original image dataset is not included in this repository because of its size and the presence of identifiable people.

Processed landmark features and experiment outputs are included for reproducibility.

## Model Comparison

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Decision Tree | 75.15% | 0.7278 |
| Random Forest | 85.01% | 0.8310 |
| Logistic Regression | 90.93% | 0.8855 |

Logistic Regression was selected using validation Macro F1.

## Final Test Results

| Metric | Value |
|---|---:|
| Accuracy | 77.58% |
| Macro Precision | 0.7908 |
| Macro Recall | 0.7837 |
| Macro F1 | 0.7672 |
| Weighted F1 | 0.7715 |

These results measure classification performance on accepted static-image features. Webcam behavior may differ because of lighting, camera quality, background, orientation, and signer variation.

## Project Structure

```text
sign_language_recognition/
├── data/
│   └── processed/
├── models/
│   ├── hand_landmarker.task
│   └── sign_classifier.joblib
├── outputs/
│   ├── evaluation/
│   └── training/
├── 01_explore_data.py
├── 02_inspect_hand.py
├── 03_extract_features.py
├── 04_review_extraction.py
├── 05_test_padding.py
├── 06_extract_padded_features.py
├── 07_prepare_splits.py
├── 08_train_models.py
├── 09_evaluate_model.py
├── 10_predict_image.py
├── 11_webcam.py
├── preprocessing.py
├── inference.py
├── model_common.py
├── run_from_07.py
└── requirements.txt
```

## Installation

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Training and Evaluation

The repository includes processed landmark features, so the pipeline can resume from step 07:

```powershell
python .\run_from_07.py
```

This runs:

- Dataset cleaning and grouped splitting
- Model training and validation comparison
- Final test evaluation

## Predict One Image

```powershell
python .\10_predict_image.py "path\to\image.jpg" --show
```

## Run Real-Time Webcam Recognition

```powershell
python .\11_webcam.py --camera 0
```

Alternative Windows backend:

```powershell
python .\11_webcam.py --camera 0 --backend dshow
```

Webcam controls:

- `Q` or `Esc`: Quit
- `M`: Toggle mirrored display
- `R`: Reset prediction smoothing

## Important Notes

- The classifier recognizes static hand shapes, not continuous sign-language sentences.
- Model scores are not calibrated guarantees of correctness.
- The webcam should contain one clear hand with good lighting.
- The raw dataset must be placed inside `data/raw/Gesture Image Data` to rerun steps 01-06.

## Author

Ziad Mostafa
