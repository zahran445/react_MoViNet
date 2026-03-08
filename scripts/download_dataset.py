"""
SAWN Dataset Downloader
Downloads the SAWN dataset from GitHub and prepares the directory structure.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://github.com/Sawn24/Sawn_Dataset/raw/main"
FILES = [
    "Traininng_PedestrianLittering.zip",
    "Traininng_VehicleLittering.zip",
    "Test_PedestrianLittering.zip",
    "Test_VehicleLittering.zip",
]
DATA_DIR = Path("data")


# ── Helpers ──────────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path):
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip archive."""
    print(f"Extracting {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    zip_path.unlink()  # remove zip after extraction


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SAWN Dataset Downloader")
    print("=" * 60)

    # Map zip names to structured folders
    mapping = {
        "Traininng_PedestrianLittering.zip": DATA_DIR / "train" / "PedestrianLittering",
        "Traininng_VehicleLittering.zip":    DATA_DIR / "train" / "VehicleLittering",
        "Test_PedestrianLittering.zip":      DATA_DIR / "test"  / "PedestrianLittering",
        "Test_VehicleLittering.zip":         DATA_DIR / "test"  / "VehicleLittering",
    }

    for filename, dest_folder in mapping.items():
        zip_dest = DATA_DIR / "zips" / filename
        url = f"{BASE_URL}/{filename}"

        if dest_folder.exists() and any(dest_folder.iterdir()):
            print(f"[SKIP] {filename} already extracted.")
            continue

        print(f"\n[DOWNLOAD] {filename}")
        download_file(url, zip_dest)
        dest_folder.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_dest, dest_folder)

    # Clean up zips folder if empty
    zips_dir = DATA_DIR / "zips"
    if zips_dir.exists() and not any(zips_dir.iterdir()):
        zips_dir.rmdir()

    print("\n[DONE] Dataset ready:")
    for split in ("train", "test"):
        for cls in ("PedestrianLittering", "VehicleLittering"):
            folder = DATA_DIR / split / cls
            if folder.exists():
                count = len(list(folder.glob("*")))
                print(f"  {split}/{cls}: {count} files")


if __name__ == "__main__":
    main()
