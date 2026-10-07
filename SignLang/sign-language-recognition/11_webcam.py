"""Live classification. Q/Esc: quit. M: mirror display only. R: reset smoothing."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from inference import PredictionSmoother, SignRecognizer, draw_landmarks, status_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Display threshold only, not calibrated or tuned on test data.")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.window < 1:
        parser.error("threshold must be in [0,1], and window must be positive.")
    import cv2
    recognizer = SignRecognizer(args.project_root.resolve())
    capture = None
    try:
        backend = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[args.backend]
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera. Close other camera apps, check Windows permissions, or try --camera 1 / --backend dshow.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        smoother = PredictionSmoother(args.window, min(10, args.window))
        mirrored = not args.no_mirror
        window_name = "Sign language recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        fps = 0.0
        print(f"Model: {recognizer.bundle['model_name']}")
        print("Q/Esc: quit | M: mirror display only | R: reset smoothing")
        print("Hold one hand in view. The current classifier covers the dataset's 12 VSL labels.")
        while True:
            started = perf_counter()
            success, frame = capture.read()
            if not success:
                raise RuntimeError("The camera opened but did not return a frame.")
            result = recognizer.process(frame)  # Raw, unmirrored input, exactly as captured.
            mean_scores = smoother.update(result.probabilities if result.status == "ok" else None)
            frame = draw_landmarks(frame, result.points_pixels)
            if mirrored:
                frame = cv2.flip(frame, 1)  # Visual aid only. Never enters MediaPipe or the classifier.

            color = (60, 210, 250)
            title = status_message(result.status)
            detail = ""
            if result.status == "ok":
                detail = f"Raw guess: {result.label}  |  Model score: {result.score:.2f}"
                if mean_scores is None:
                    title = "Hold the sign steady..."
                else:
                    best = int(mean_scores.argmax())
                    label = str(recognizer.classes[best])
                    score = float(mean_scores[best])
                    if score >= args.threshold:
                        title = f"Gesture: {label}  |  Smoothed score: {score:.2f}"
                        color = (80, 230, 110)
                    else:
                        title = f"Uncertain  |  Best guess: {label} ({score:.2f})"

            # Put the HUD outside the image to avoid covering the hand.
            width = max(680, frame.shape[1])
            canvas = np.zeros((frame.shape[0] + 120, width, 3), dtype=np.uint8)
            canvas[95:95 + frame.shape[0], :frame.shape[1]] = frame
            cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, detail, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"FPS: {fps:.1f} | Q: quit | M: mirror view | R: reset", (12, 81),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 175, 175), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Scores are model outputs; an unfamiliar gesture can still get a label.",
                        (12, canvas.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            elapsed = perf_counter() - started
            current_fps = 1.0 / max(elapsed, 1e-6)
            fps = current_fps if fps == 0 else 0.85 * fps + 0.15 * current_fps
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("m"), ord("M")):
                mirrored = not mirrored
            if key in (ord("r"), ord("R")):
                smoother.reset()
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break  # Some backends destroy the window before reporting visibility.
    finally:
        if capture is not None:
            capture.release()
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
