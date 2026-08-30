"""Download WildFake from ModelScope into this project's data layout.

Writes data/<split>/real and data/<split>/ai, which is exactly what
datasets.RealAIDataset (training) and evaluate.TestImageDataset expect, so
train.py needs no changes afterwards.

Every candidate image is checked against the held-out validation set built by
holdout_index.py and dropped if it matches -- see that file for why filename
checks are not enough. Running WITHOUT --exclude-index is allowed but warns
loudly, because contaminating training data with the benchmark invalidates
every number you report from it.

WildFake's column names are not assumed. Run --inspect first to see the real
schema, then pass --image-column / --label-column if the auto-detection picks
wrong.

Usage:
    python prepare_wildfake.py --inspect
    python prepare_wildfake.py --exclude-index holdout_index.json
    python prepare_wildfake.py --exclude-index holdout_index.json --limit 20000
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageFile

from holdout_index import HoldoutFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Candidate column names, most specific first. WildFake's schema is not
# documented here, so guess from the usual conventions and let --inspect and
# the explicit flags settle it when the guess is wrong.
IMAGE_COLUMNS = ["image", "img", "image:FILE", "picture", "file", "path", "image_path"]
LABEL_COLUMNS = ["label", "labels", "target", "class", "is_fake", "fake", "category"]

# Label values that mean "AI-generated". Everything else is treated as real.
AI_VALUES = {1, "1", "fake", "ai", "aigc", "synthetic", "generated", "true", True}
REAL_VALUES = {0, "0", "real", "authentic", "nature", "natural", "false", False}


def pick_column(candidates, available, override, kind):
    """Resolve which dataset column to use, preferring an explicit override."""
    if override:
        if override not in available:
            raise SystemExit(
                f"--{kind}-column '{override}' not in dataset columns: {sorted(available)}")
        return override
    for name in candidates:
        if name in available:
            return name
    raise SystemExit(
        f"Could not auto-detect the {kind} column. Columns are: {sorted(available)}\n"
        f"Re-run with --{kind}-column <name>.")


def to_pil(value):
    """Coerce whatever the dataset yields for an image into a PIL image."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (str, Path)):
        return Image.open(value)
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    if isinstance(value, dict):
        # HuggingFace-style {"bytes": ..., "path": ...}
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise TypeError(f"Unrecognised image value of type {type(value)}")


def to_class(value):
    """Map a dataset label onto 'ai' or 'real', or None when unrecognised."""
    if value in AI_VALUES:
        return "ai"
    if value in REAL_VALUES:
        return "real"
    if isinstance(value, str):
        low = value.strip().lower()
        if low in AI_VALUES:
            return "ai"
        if low in REAL_VALUES:
            return "real"
    return None


def import_msdataset():
    """Import MsDataset with this project's own directory off sys.path.

    modelscope.msdatasets does `from datasets import Dataset, DatasetDict,
    Features, IterableDataset` -- meaning the HuggingFace `datasets` package.
    This repo contains its own datasets.py (the real-vs-AI image loader) in
    the same directory, and Python searches the script's directory FIRST, so
    that import resolves to the project's module and fails on every name
    modelscope expects.

    Dropping the project directory for the duration of this one import is the
    contained fix. The permanent fix is to rename datasets.py to something
    that cannot collide -- see the note in the README.
    """
    here = Path(__file__).resolve().parent
    saved_path = list(sys.path)
    # If it was already imported (it should not have been), set it aside so
    # the real package gets a clean shot at the name.
    saved_module = sys.modules.pop("datasets", None)

    def is_project_dir(entry):
        if entry in ("", "."):
            return True
        try:
            return Path(entry).resolve() == here
        except (OSError, ValueError):
            return False

    sys.path[:] = [p for p in sys.path if not is_project_dir(p)]
    try:
        from modelscope.msdatasets import MsDataset
        return MsDataset
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "modelscope":
            raise SystemExit(
                "modelscope is not installed. It is needed only for this "
                "dataset-preparation script:\n    pip install modelscope") from exc
        raise
    finally:
        sys.path[:] = saved_path
        if saved_module is not None:
            sys.modules["datasets"] = saved_module


def load_filter(path):
    """Load the holdout filter, or None when exclusion is explicitly disabled."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"No holdout index at {p}. Build one first:\n"
            f"  python holdout_index.py --add <coco_val2017> --add <dalle_advanced>")
    f = HoldoutFilter(p)
    print(f"Loaded holdout filter: {len(f)} exact hashes, {len(f.hashes)} perceptual, "
          f"matching within {f.max_distance} bits")
    return f


def inspect(ds):
    """Print the dataset's real schema and one sample, then stop."""
    row = ds[0]
    print()
    print("Columns:")
    for k, v in row.items():
        shown = type(v).__name__
        if isinstance(v, (str, int, float, bool)):
            shown += f" = {v!r}"
        print(f"  {k:<20} {shown}")
    print()
    print("Distinct values in likely label columns (first 500 rows):")
    for name in row:
        if name in LABEL_COLUMNS or "label" in name.lower():
            values = {ds[i][name] for i in range(min(500, len(ds)))}
            print(f"  {name}: {sorted(values, key=str)[:10]}")
    print()
    print("Re-run without --inspect, adding --image-column / --label-column "
          "if the auto-detection above looks wrong.")


def main():
    ap = argparse.ArgumentParser(description="Prepare WildFake into data/<split>/{real,ai}.")
    ap.add_argument("--dataset", default="hy2628982280/WildFake")
    ap.add_argument("--subset", default="default")
    ap.add_argument("--dataset-split", default="train",
                    help="Split to pull FROM the source dataset")
    ap.add_argument("--split", default="train",
                    help="Split to write INTO, i.e. data/<split>/{real,ai}")
    ap.add_argument("--out", default="data", help="Output root (default: data)")
    ap.add_argument("--exclude-index", default="holdout_index.json",
                    help="Index from holdout_index.py. Pass '' to disable (not advised).")
    ap.add_argument("--image-column", default=None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after writing this many images per class")
    ap.add_argument("--inspect", action="store_true",
                    help="Print the dataset schema and exit without writing anything")
    a = ap.parse_args()

    MsDataset = import_msdataset()

    print(f"Loading {a.dataset} (subset={a.subset}, split={a.dataset_split})...")
    ds = MsDataset.load(a.dataset, subset_name=a.subset, split=a.dataset_split)
    print(f"Loaded {len(ds)} rows")

    if a.inspect:
        inspect(ds)
        return

    holdout = load_filter(a.exclude_index)
    if holdout is None:
        print()
        print("!! WARNING: no holdout index loaded. Training data will NOT be")
        print("!! checked against the benchmark set. Any score you report from")
        print("!! that benchmark afterwards is meaningless. Ctrl-C to stop.")
        print()

    columns = set(ds[0].keys())
    img_col = pick_column(IMAGE_COLUMNS, columns, a.image_column, "image")
    lbl_col = pick_column(LABEL_COLUMNS, columns, a.label_column, "label")
    print(f"Using image column '{img_col}', label column '{lbl_col}'")

    out_root = Path(a.out) / a.split
    for cls in ("real", "ai"):
        (out_root / cls).mkdir(parents=True, exist_ok=True)

    written = {"real": 0, "ai": 0}
    excluded_exact = excluded_perceptual = 0
    unreadable = unlabelled = 0

    for i in range(len(ds)):
        if a.limit and written["real"] >= a.limit and written["ai"] >= a.limit:
            break

        row = ds[i]
        cls = to_class(row[lbl_col])
        if cls is None:
            unlabelled += 1
            continue
        if a.limit and written[cls] >= a.limit:
            continue

        try:
            img = to_pil(row[img_col]).convert("RGB")
        except Exception:
            unreadable += 1
            continue

        # Re-encode to a stable byte form so the sha256 is comparable with the
        # index, which hashed files as they sat on disk.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        raw = buf.getvalue()

        if holdout is not None:
            blocked, reason, source = holdout.check(raw, img)
            if blocked:
                # Same picture, possibly re-encoded or resized -- exactly the
                # case a filename or byte comparison would let through.
                if reason == "exact":
                    excluded_exact += 1
                else:
                    excluded_perceptual += 1
                    # Show the first few so the exclusions can be sanity-checked
                    # rather than taken on trust.
                    if excluded_perceptual <= 5:
                        print(f"  excluded {reason}: matches {source}")
                continue

        name = f"wildfake_{i:08d}.jpg"
        (out_root / cls / name).write_bytes(raw)
        written[cls] += 1

        total = written["real"] + written["ai"]
        if total and total % 1000 == 0:
            print(f"  {total} written (real={written['real']}, ai={written['ai']}), "
                  f"{excluded_exact + excluded_perceptual} excluded")

    print()
    print(f"Wrote to {out_root}")
    print(f"  real : {written['real']}")
    print(f"  ai   : {written['ai']}")
    print(f"Excluded as held-out benchmark data:")
    print(f"  exact byte match      : {excluded_exact}")
    print(f"  perceptual match      : {excluded_perceptual}")
    if unlabelled:
        print(f"Skipped {unlabelled} rows with an unrecognised label "
              f"(check --label-column and AI_VALUES/REAL_VALUES)")
    if unreadable:
        print(f"Skipped {unreadable} unreadable images")


if __name__ == "__main__":
    main()
