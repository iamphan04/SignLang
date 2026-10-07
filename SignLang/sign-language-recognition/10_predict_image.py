"""Predict a single image using the same path as webcam frames."""

import argparse
from pathlib import Path

import numpy as np

from inference import SignRecognizer, draw_landmarks, status_message
from model_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image path; quote it if it contains spaces.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--show", action="store_true", help="Also open the preview window.")
    args = parser.parse_args()
    import cv2
    root = args.project_root.resolve()
    image_path = args.image.resolve()
    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    recognizer = SignRecognizer(root)
    try:
        result = recognizer.process(image)
        record = {"image": str(image_path), "model": recognizer.bundle["model_name"],
                  "status": result.status, "detected_hands": result.detected_hands}
        print(f"Status: {result.status} | {status_message(result.status)}")
        preview = draw_landmarks(image, result.points_pixels)
        text = status_message(result.status)
        if result.status == "ok":
            top = np.argsort(result.probabilities)[-3:][::-1]
            record["prediction"] = result.label
            record["top_3"] = [{"label": str(recognizer.classes[i]), "model_score": float(result.probabilities[i])} for i in top]
            text = f"Prediction: {result.label} | Model score: {result.score:.2f}"
            print(text)
            for item in record["top_3"]:
                print(f"  {item['label']}: {item['model_score']:.4f}")
        # Scale display only AFTER inference, so features stay on the original image.
        scale = min(1.0, 1000 / max(preview.shape[:2]))
        if scale < 1:
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        canvas_width = max(650, preview.shape[1])
        canvas = np.zeros((preview.shape[0] + 65, canvas_width, 3), dtype=np.uint8)
        canvas[65:, :preview.shape[1]] = preview
        cv2.putText(canvas, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        output = root / "outputs" / "prediction"
        output.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".png", canvas)
        if not success:
            raise RuntimeError("Could not encode prediction preview.")
        (output / "prediction.png").write_bytes(encoded.tobytes())
        write_json(output / "prediction.json", record)
        print(f"Saved preview and prediction: {output}")
        print("Model scores are not calibrated guarantees of correctness.")
        if args.show:
            cv2.imshow("Single-image prediction", canvas)
            cv2.waitKey(0)
    finally:
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
