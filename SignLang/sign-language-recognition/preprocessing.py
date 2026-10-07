"""Shared 42-feature preprocessing for training and webcam prediction.

Use add_image_padding() before detection and normalize_landmarks() afterwards
in the padded extraction and webcam pipelines. The original extraction script
can still import normalize_landmarks() alone for baseline comparisons.
Run this file directly to see a worked example using the rounded coordinates
reported for the user's first B image (750 pixels wide, 1334 pixels high).
The example coordinates are used ONLY by the demonstration, never by the
reusable normalization function.
"""

import numpy as np


IMAGE_PADDING_FRACTION = 0.25


def add_image_padding(image_rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return a black-padded RGB image plus its left/top offsets in pixels.

    Add 25% of the original width to EACH horizontal side and 25% of the
    height to EACH vertical side, rounding to whole pixels. Source pixels
    remain unchanged. For example, 400 x 400 becomes 600 x 600.
    The returned offsets allow webcam overlays to map back to the raw frame.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image with shape (height, width, 3).")
    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("The image must have positive width and height.")
    pad_x = max(1, round(width * IMAGE_PADDING_FRACTION))
    pad_y = max(1, round(height * IMAGE_PADDING_FRACTION))
    padded = np.pad(image_rgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
                    mode="constant", constant_values=0)
    return padded, pad_x, pad_y


def normalize_landmarks(
    landmarks_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert one hand's image-relative x/y coordinates into 42 features.

    Input: an array of shape (21, 2), ordered by MediaPipe landmark ID.
    Supply the dimensions of the SAME image passed to the hand detector.
    Output: a float32 array of shape (42,), ordered x0,y0,...,x20,y20.

    Method: convert to pixel units, subtract wrist 0, then divide by the
    Euclidean distance from wrist 0 to middle-finger MCP 9. A single positive
    scale is used for both axes. No rotation, mirroring, or clipping is applied.
    These are per-image geometric operations; no dataset statistics are fitted.
    """
    points_xy = np.asarray(landmarks_xy, dtype=np.float64)
    dimensions = np.array([image_width, image_height], dtype=np.float64)

    if points_xy.shape != (21, 2):
        raise ValueError("Expected 21 landmarks, each with x and y: shape (21, 2).")
    if not np.isfinite(points_xy).all():
        raise ValueError("Landmark coordinates must be finite numbers.")
    if not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("Image width and height must be finite positive numbers.")

    # Use the same units on both axes, including for non-square images.
    points_pixels = points_xy * dimensions

    # Position normalization: the wrist becomes the origin (0, 0).
    centered = points_pixels - points_pixels[0]

    # Scale normalization: the wrist-to-middle-MCP distance becomes 1.
    scale = np.linalg.norm(centered[9])
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("The wrist-to-middle-finger reference distance is too small.")

    normalized = centered / scale
    return normalized.reshape(42).astype(np.float32)


def _run_demo() -> None:
    """Demonstrate the function on the user's reported single-image result."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    width, height = 750, 1334
    example_xy = np.array([
        [0.119960, 0.818826],
        [0.219395, 0.792907],
        [0.294395, 0.740464],
        [0.250901, 0.693110],
        [0.156675, 0.678578],
        [0.276765, 0.643622],
        [0.289569, 0.573498],
        [0.291909, 0.530694],
        [0.292313, 0.492197],
        [0.212417, 0.630652],
        [0.219687, 0.547863],
        [0.222398, 0.499953],
        [0.223231, 0.459523],
        [0.150801, 0.636619],
        [0.153067, 0.558238],
        [0.152288, 0.512087],
        [0.152303, 0.471949],
        [0.084934, 0.657106],
        [0.077902, 0.599629],
        [0.069908, 0.564519],
        [0.062259, 0.531963],
    ], dtype=np.float64)

    dimensions = np.array([width, height])
    pixels = example_xy * dimensions
    centered = pixels - pixels[0]
    scale = np.linalg.norm(centered[9])
    features = normalize_landmarks(example_xy, width, height)
    normalized = features.reshape(21, 2)

    print("Worked example: rounded coordinates from your B image (750 x 1334).")
    print("This demonstration does not run MediaPipe or classify an image.")
    print(f"\nWrist pixel coordinates: {pixels[0]}")
    print(f"Reference distance (wrist 0 to middle-finger base 9): {scale:.6f} px")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature data type: {features.dtype}")
    print(f"Normalized wrist: {normalized[0]}")
    print(f"Normalized reference length: {np.linalg.norm(normalized[9]):.6f}")
    print(f"Normalized index fingertip (8): {normalized[8]}")

    print("\nID       x_normalized       y_normalized")
    for index, (x, y) in enumerate(normalized):
        print(f"{index:2d}       {x:12.6f}       {y:12.6f}")

    # Check changes of position and uniform scale on the existing coordinates.
    # These are geometric checks, not new images or training augmentation.
    shifted_xy = (pixels + np.array([120.0, -80.0])) / dimensions
    scaled_xy = (pixels * 0.6 + np.array([250.0, 100.0])) / dimensions
    new_canvas = np.array([1500, 1500])
    new_canvas_xy = pixels / new_canvas

    cases = [
        ("Translation", shifted_xy, width, height),
        ("Uniform scaling and translation", scaled_xy, width, height),
        ("Same hand on a differently sized canvas", new_canvas_xy, 1500, 1500),
    ]
    print("\nGeometric invariance checks:")
    for name, changed_xy, changed_width, changed_height in cases:
        changed_features = normalize_landmarks(changed_xy, changed_width, changed_height)
        same = np.allclose(features, changed_features, rtol=1e-6, atol=1e-6)
        difference = np.max(np.abs(features - changed_features))
        print(f"{name}: {'PASS' if same else 'FAIL'} | max difference: {difference:.3e}")
        if not same:
            raise AssertionError(f"Normalization check failed: {name}")

    # Draw the same geometry before and after each normalization step.
    chains = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
        [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [17, 0],
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 5.8))
    stages = [
        (pixels, "1. Image coordinates", "Pixels"),
        (centered, "2. Position normalization", "Pixels relative to wrist"),
        (normalized, "3. Position + scale", "Reference lengths"),
    ]
    for axis, (points, title, units) in zip(axes, stages):
        for chain in chains:
            axis.plot(points[chain, 0], points[chain, 1], color="#158578", linewidth=2)
        axis.scatter(points[:, 0], points[:, 1], s=19, color="#158578", zorder=3)
        axis.plot(points[[0, 9], 0], points[[0, 9], 1], "--", color="#d36c28", linewidth=2)
        axis.scatter(points[[0, 9], 0], points[[0, 9], 1], s=42, color="#d36c28", zorder=4)
        for index, text in [(0, "0: wrist"), (9, "9: reference")]:
            axis.annotate(text, points[index], xytext=(8, 0), textcoords="offset points", fontsize=8)
        axis.set_title(title, fontsize=11, fontweight="bold", pad=13)
        axis.set_xlabel(f"X ({units})", fontsize=8)
        axis.set_ylabel(f"Y ({units})", fontsize=8)
        axis.grid(alpha=0.18)
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.margins(x=0.5, y=0.15)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    figure.suptitle("The same hand, expressed relative to its wrist and palm size", fontsize=13)
    figure.text(0.5, 0.02, "Orange dashed segment: reference distance from landmark 0 to landmark 9.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "normalization_example.png"
    figure.savefig(figure_path, dpi=160)
    print(f"\nSaved diagram: {figure_path}")
    print("Close the diagram window to finish.")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    _run_demo()
