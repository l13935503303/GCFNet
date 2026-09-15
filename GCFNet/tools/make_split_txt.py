import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate train.txt and val.txt for YESeg-style datasets."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getenv("YESEG_ROOT", ""),
        help="Dataset root path. Defaults to YESEG_ROOT.",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="label",
        help="Label directory under root.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".png",
        help="Label suffix, e.g. .png",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=347,
        help="Random seed.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stratified",
        choices=["random", "stratified"],
        help="Split mode.",
    )
    parser.add_argument(
        "--ignore-labels",
        type=str,
        default="0,255",
        help="Comma-separated labels ignored when extracting dominant class in stratified mode.",
    )
    return parser.parse_args()


def parse_int_set(csv_text: str):
    out = set()
    for part in str(csv_text).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def list_label_keys(label_dir: Path, suffix: str):
    keys = []
    for p in sorted(label_dir.rglob(f"*{suffix}")):
        if p.is_file():
            keys.append(p.relative_to(label_dir).with_suffix("").as_posix())
    return keys


def write_split_files(root: Path, train_keys, val_keys):
    train_path = root / "train.txt"
    val_path = root / "val.txt"
    train_path.write_text("\n".join(train_keys), encoding="utf-8")
    val_path.write_text("\n".join(val_keys), encoding="utf-8")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


def random_split(keys, train_ratio, seed):
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    n_train = int(len(keys) * float(train_ratio))
    n_train = max(1, min(n_train, len(keys) - 1)) if len(keys) > 1 else len(keys)
    return keys[:n_train], keys[n_train:]


def read_label_array(label_path: Path):
    if Image is None:
        raise ImportError("Please install pillow first: pip install pillow")
    with Image.open(label_path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        # Keep compatibility with possible pseudo-color masks.
        arr = arr[..., 0]
    return arr


def dominant_class(label_path: Path, ignore_labels):
    arr = read_label_array(label_path)
    flat = arr.reshape(-1)
    if ignore_labels:
        mask = np.ones_like(flat, dtype=bool)
        for ig in ignore_labels:
            mask &= flat != ig
        flat = flat[mask]
    if flat.size == 0:
        return None
    values, counts = np.unique(flat, return_counts=True)
    return int(values[np.argmax(counts)])


def stratified_split(label_dir: Path, keys, suffix: str, train_ratio, seed, ignore_labels):
    rng = random.Random(seed)
    groups = defaultdict(list)
    skipped = []

    for key in keys:
        label_path = label_dir / f"{key}{suffix}"
        cls = dominant_class(label_path, ignore_labels=ignore_labels)
        if cls is None:
            skipped.append(key)
            continue
        groups[cls].append(key)

    train_keys = []
    val_keys = []
    for cls in sorted(groups.keys()):
        items = groups[cls]
        rng.shuffle(items)
        if len(items) == 1:
            train_keys.extend(items)
            continue

        n_train = int(len(items) * float(train_ratio))
        n_train = max(1, min(n_train, len(items) - 1))
        train_keys.extend(items[:n_train])
        val_keys.extend(items[n_train:])

    train_keys = sorted(train_keys)
    val_keys = sorted(val_keys)

    if len(val_keys) == 0 and len(train_keys) > 1:
        # Safety fallback: keep at least one validation sample.
        val_keys.append(train_keys[-1])
        train_keys = train_keys[:-1]

    return train_keys, val_keys, skipped, groups


def main():
    args = parse_args()
    if not str(args.root).strip():
        raise ValueError("Dataset root is empty. Pass --root or set YESEG_ROOT.")

    root = Path(args.root)
    label_dir = root / args.label_dir

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got: {args.train_ratio}")

    keys = list_label_keys(label_dir, args.suffix)
    if len(keys) == 0:
        raise RuntimeError(f"No labels found in: {label_dir} (suffix={args.suffix})")

    if args.mode == "random":
        train_keys, val_keys = random_split(keys, args.train_ratio, args.seed)
        print("[Split mode] random")
        print(f"Total: {len(keys)} | Train: {len(train_keys)} | Val: {len(val_keys)}")
    else:
        ignore_labels = parse_int_set(args.ignore_labels)
        train_keys, val_keys, skipped, groups = stratified_split(
            label_dir=label_dir,
            keys=keys,
            suffix=args.suffix,
            train_ratio=args.train_ratio,
            seed=args.seed,
            ignore_labels=ignore_labels,
        )
        print("[Split mode] stratified (dominant class)")
        print(
            f"Total: {len(keys)} | Used: {len(train_keys) + len(val_keys)} | "
            f"Skipped(all ignored): {len(skipped)}"
        )
        print(f"Train: {len(train_keys)} | Val: {len(val_keys)} | Groups: {len(groups)}")

    write_split_files(root, train_keys, val_keys)


if __name__ == "__main__":
    main()
