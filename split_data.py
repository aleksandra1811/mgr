"""
Podział danych CelebA-HQ na zbiory train / val / test.

Domyślny podział: 80 % train / 10 % val / 10 % test
Dla 30 000 obrazów: 24 000 / 3 000 / 3 000

Użycie:
    python split_data.py --source ../archive/celeba_hq_256 --dest ./data
"""
import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_data(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """
    Podziel pliki z source_dir na train/val/test i utwórz symlinki w dest_dir.

    Args:
        source_dir: Katalog źródłowy z obrazami.
        dest_dir:   Katalog docelowy; zostaną utworzone podfoldery train/val/test.
        train_ratio: Ułamek danych treningowych.
        val_ratio:   Ułamek danych walidacyjnych; reszta trafia do test.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    images = [f for f in source_path.iterdir()
              if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]

    if not images:
        print(f"Brak obrazów w {source_dir}!")
        return

    print(f"Znaleziono {len(images)} obrazów w {source_dir}")

    random.seed(42)
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:],
    }

    print(f"\nPodział:")
    for name, subset in splits.items():
        print(f"  {name:<6}: {len(subset):>5} obrazów  ({len(subset) / n * 100:.1f} %)")

    for name in splits:
        (dest_path / name).mkdir(parents=True, exist_ok=True)

    print("\nTworzenie symlinków...")
    for split_name, split_images in splits.items():
        split_dir = dest_path / split_name
        for img_path in tqdm(split_images, desc=f"  {split_name}"):
            dest_file = split_dir / img_path.name
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            try:
                dest_file.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dest_file)

    print(f"\nGotowe! Dane w: {dest_path}")
    print(f"  {dest_path}/")
    print(f"    ├── train/ ({len(splits['train'])} plików)")
    print(f"    ├── val/   ({len(splits['val'])} plików)")
    print(f"    └── test/  ({len(splits['test'])} plików)")


def main():
    parser = argparse.ArgumentParser(
        description="Podział danych CelebA-HQ na train/val/test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",      type=str,   default="../archive/celeba_hq_256")
    parser.add_argument("--dest",        type=str,   default="./data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    args = parser.parse_args()

    split_data(
        source_dir=args.source,
        dest_dir=args.dest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
