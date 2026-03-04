import os
import shutil
import subprocess
from pathlib import Path


DOWNLOAD_ROOT = Path("/downloads")
STAGING_DIR = DOWNLOAD_ROOT / "_staging"
TARGET_FILE = DOWNLOAD_ROOT / "dataset.csv"
DEFAULT_DATASET = "shivamb/netflix-shows"


def require_credentials() -> None:
    missing = [
        name
        for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(
            "Missing Kaggle credentials: "
            + ", ".join(missing)
            + ". Set them in your shell before running the helper container."
        )


def main() -> None:
    require_credentials()

    dataset_ref = os.getenv("KAGGLE_DATASET", DEFAULT_DATASET)
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_ref,
            "-p",
            str(STAGING_DIR),
            "--unzip",
        ],
        check=True,
    )

    csv_files = sorted(
        STAGING_DIR.rglob("*.csv"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not csv_files:
        raise SystemExit(f"No CSV files were extracted from {dataset_ref}.")

    if TARGET_FILE.exists():
        TARGET_FILE.unlink()

    shutil.copyfile(csv_files[0], TARGET_FILE)
    shutil.rmtree(STAGING_DIR)

    print(f"Downloaded {dataset_ref}")
    print(f"Saved {csv_files[0].name} as {TARGET_FILE}")


if __name__ == "__main__":
    main()
