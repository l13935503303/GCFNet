import argparse
import random
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".pt", ".pth"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate train_pohang.txt and val_pohang.txt with explicit, "
            "leakage-aware Pohang splits."
        )
    )
    parser.add_argument("--root", required=True, help="Extracted Pohang dataset root.")
    parser.add_argument(
        "--mode",
        default="scene",
        choices=["scene", "block", "random"],
        help=(
            "scene: train on --train-scene and validate on --val-scene; "
            "block: split contiguous numeric tile blocks inside each scene; "
            "random: sample-level random split, kept only for diagnostics."
        ),
    )
    parser.add_argument("--scenes", default="cloud,no_cloud", help="Comma-separated scenes for block/random mode.")
    parser.add_argument("--train-scene", default="cloud", help="Training scene for scene mode.")
    parser.add_argument("--val-scene", default="no_cloud", help="Validation scene for scene mode.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for block/random mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--block-size", type=int, default=32, help="Tiles per contiguous block in block mode.")
    parser.add_argument(
        "--gap-blocks",
        type=int,
        default=1,
        help="Neighbouring blocks removed from train around validation blocks in block mode.",
    )
    parser.add_argument("--train-file", default="train_pohang.txt")
    parser.add_argument("--val-file", default="val_pohang.txt")
    return parser.parse_args()


def numeric_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def list_stems(folder: Path):
    stems = set()
    if not folder.exists():
        return stems
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stems.add(p.relative_to(folder).with_suffix("").as_posix())
    return stems


def list_scene_keys(root: Path, scene: str):
    scene_root = root / scene
    opt = list_stems(scene_root / "GF2")
    sar = list_stems(scene_root / "GF3")
    label = list_stems(scene_root / "label")
    stems = sorted(opt & sar & label, key=lambda s: numeric_sort_key(Path(s)))
    return [f"{scene}/GF2/{stem}" for stem in stems]


def read_scene_list(text: str):
    return [s.strip() for s in str(text).split(",") if s.strip()]


def split_scene(root: Path, train_scene: str, val_scene: str):
    train_keys = list_scene_keys(root, train_scene)
    val_keys = list_scene_keys(root, val_scene)
    return train_keys, val_keys


def split_random(all_keys, val_ratio: float, seed: int):
    keys = list(all_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = int(round(len(keys) * float(val_ratio)))
    n_val = max(1, min(n_val, len(keys) - 1)) if len(keys) > 1 else 0
    val = sorted(keys[:n_val])
    train = sorted(keys[n_val:])
    return train, val


def split_blocks_by_scene(root: Path, scenes, val_ratio: float, seed: int, block_size: int, gap_blocks: int):
    rng = random.Random(seed)
    train_keys = []
    val_keys = []
    dropped_gap_keys = []

    block_size = max(1, int(block_size))
    gap_blocks = max(0, int(gap_blocks))

    for scene in scenes:
        keys = list_scene_keys(root, scene)
        if len(keys) <= 1:
            train_keys.extend(keys)
            continue

        blocks = [keys[i:i + block_size] for i in range(0, len(keys), block_size)]
        block_ids = list(range(len(blocks)))
        rng.shuffle(block_ids)
        n_val = int(round(len(blocks) * float(val_ratio)))
        n_val = max(1, min(n_val, len(blocks) - 1))
        val_blocks = set(block_ids[:n_val])
        gap = set()
        for b in val_blocks:
            for nb in range(b - gap_blocks, b + gap_blocks + 1):
                if 0 <= nb < len(blocks) and nb not in val_blocks:
                    gap.add(nb)

        for i, block in enumerate(blocks):
            if i in val_blocks:
                val_keys.extend(block)
            elif i in gap:
                dropped_gap_keys.extend(block)
            else:
                train_keys.extend(block)

    return sorted(train_keys), sorted(val_keys), sorted(dropped_gap_keys)


def check_disjoint(train_keys, val_keys):
    overlap = sorted(set(train_keys) & set(val_keys))
    if overlap:
        raise RuntimeError(f"Split overlap detected: {overlap[:10]}")


def write_lines(path: Path, keys):
    path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Pohang root not found: {root}")

    if args.mode == "scene":
        train_keys, val_keys = split_scene(root, args.train_scene, args.val_scene)
        dropped = []
    elif args.mode == "block":
        scenes = read_scene_list(args.scenes)
        train_keys, val_keys, dropped = split_blocks_by_scene(
            root=root,
            scenes=scenes,
            val_ratio=args.val_ratio,
            seed=args.seed,
            block_size=args.block_size,
            gap_blocks=args.gap_blocks,
        )
    else:
        scenes = read_scene_list(args.scenes)
        all_keys = []
        for scene in scenes:
            all_keys.extend(list_scene_keys(root, scene))
        train_keys, val_keys = split_random(all_keys, args.val_ratio, args.seed)
        dropped = []

    if len(train_keys) == 0 or len(val_keys) == 0:
        raise RuntimeError(f"Empty split: train={len(train_keys)}, val={len(val_keys)}")

    check_disjoint(train_keys, val_keys)
    train_path = root / args.train_file
    val_path = root / args.val_file
    write_lines(train_path, train_keys)
    write_lines(val_path, val_keys)

    print(f"[Pohang split] mode={args.mode}")
    print(f"root={root}")
    print(f"train={len(train_keys)} -> {train_path}")
    print(f"val={len(val_keys)} -> {val_path}")
    if dropped:
        print(f"gap_dropped_from_train={len(dropped)}")
    print("overlap=0")
    print("key_format=scene/GF2/stem")
    if args.mode == "block":
        print("assumption=contiguous numeric filenames approximate spatial adjacency")


if __name__ == "__main__":
    main()
