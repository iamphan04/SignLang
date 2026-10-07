"""Resume the project at step 07. Stop at the first failing stage."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Open the webcam after 07, 08, and 09 succeed.")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    scripts = ["07_prepare_splits.py", "08_train_models.py", "09_evaluate_model.py"]
    if args.camera:
        scripts.append("11_webcam.py")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    progress = {"started_at": datetime.now(timezone.utc).isoformat(), "start_step": "07", "stages": []}
    status_path = output / "run_status.json"
    for script in scripts:
        record = {"script": script, "state": "running"}
        progress["stages"].append(record)
        status_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        print(f"\nRunning {script}", flush=True)
        command = [sys.executable, str(root / script)]
        if script == "11_webcam.py":
            command += ["--camera", str(args.camera_index)]
        result = subprocess.run(command, cwd=root)
        record.update(state="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        status_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            print(f"Stopped at {script}. Share the error above; later stages were not run.", flush=True)
            raise SystemExit(result.returncode)
    print("\nRequested stages finished successfully.")
    if not args.camera:
        print('To try the webcam: .\\.venv\\Scripts\\python.exe .\\11_webcam.py')


if __name__ == "__main__":
    main()
