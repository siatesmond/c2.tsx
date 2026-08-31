"""Extract a capped, balanced subset of WildFake images into data/{train,val}/{real,ai}.

WildFake (ModelScope hy2628982280/WildFake) ships as per-source zips. This
script pulls a random subset of images *directly* out of those zips (member by
member -- it never unpacks a whole archive, so it works on a tight disk) and
drops them alongside the existing CIFAKE data so `train.py` can train on the
mix with no code change.

  * Real sources  -> label 0 -> data/<split>/real/
  * Diffusion sources -> label 1 -> data/<split>/ai/
  * Every file is renamed  wf_<source>_<originalname>  so it is easy to spot,
    filter, or delete later, and so names never collide across sources.
  * A small fraction (--val_frac) goes to data/val/ instead of data/train/, so
    the validation split stops being CIFAKE-only.

IMPORTANT: coco.zip and DALLE.zip are deliberately NOT listed here -- the
competition's held-out demo benchmark ("a subset of WildFake: COCO val2017 +
DALL-E Advanced") is carved out of those two, so training on them would be
training on the benchmark.

Usage:
    python prep_wildfake.py --src ../wildfake_raw --per_class 40000 --val_frac 0.05
    python prep_wildfake.py --src ../wildfake_raw --per_class 40000 --delete_zips
"""

import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# (relative-path-inside-src, short source tag). label comes from the group.
REAL_ZIPS = [
    ("Images/Real/imagenet.zip", "imagenet"),
    ("Images/Real/church.zip", "church"),
    ("Images/Real/ffhq.zip", "ffhq"),
    ("Images/Real/afhq.zip", "afhq"),
    ("Images/Real/celebahq.zip", "celebahq"),
]
AI_ZIPS = [
    ("Images/Diffusion_based/DDIM.zip", "ddim"),
    ("Images/Diffusion_based/DDPM.zip", "ddpm"),
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _pick_members(src, zips, per_class, rng):
    """Return a shuffled list of (zip_path, member_name, source_tag), capped at
    per_class, drawn as evenly as possible across the given zips."""
    pools = []
    for rel, tag in zips:
        zpath = src / rel
        if not zpath.exists() or zpath.stat().st_size < 1024:
            print(f"  ! skipping {rel} (missing or still an LFS pointer)")
            continue
        with zipfile.ZipFile(zpath) as zf:
            members = [n for n in zf.namelist()
                       if n.lower().endswith(IMG_EXTS) and not n.endswith("/")]
        rng.shuffle(members)
        pools.append((zpath, tag, members))
        print(f"  {tag}: {len(members)} images available")

    if not pools:
        return []

    # Round-robin across sources so one big zip doesn't dominate the subset.
    per_source = max(1, per_class // len(pools))
    picked = []
    for zpath, tag, members in pools:
        take = members[:per_source]
        picked.extend((zpath, m, tag) for m in take)
    rng.shuffle(picked)
    return picked[:per_class]


def _extract(picked, out_train, out_val, val_frac, rng):
    """Write each picked member as a flat wf_<tag>_<name> file, re-encoded to a
    valid RGB image, into the train or val class folder."""
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    open_zips = {}
    n_ok = n_val = n_bad = 0
    try:
        for i, (zpath, member, tag) in enumerate(picked, 1):
            zf = open_zips.get(zpath) or open_zips.setdefault(zpath, zipfile.ZipFile(zpath))
            stem = Path(member).name
            dst_dir = out_val if rng.random() < val_frac else out_train
            dst = dst_dir / f"wf_{tag}_{stem}"
            if dst.suffix.lower() not in IMG_EXTS:
                dst = dst.with_suffix(".jpg")
            if dst.exists():
                n_ok += 1
                continue
            try:
                raw = zf.read(member)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                # Normalise everything to JPEG to keep the on-disk size down.
                dst = dst.with_suffix(".jpg")
                im.save(dst, format="JPEG", quality=95)
            except Exception:
                n_bad += 1
                continue
            if dst_dir is out_val:
                n_val += 1
            n_ok += 1
            if i % 2000 == 0:
                print(f"    {i}/{len(picked)} extracted ...", flush=True)
    finally:
        for zf in open_zips.values():
            zf.close()
    return n_ok, n_val, n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../wildfake_raw",
                    help="Path to the cloned WildFake repo (with Images/... zips).")
    ap.add_argument("--dst", default="data",
                    help="Dataset root; images go to <dst>/{train,val}/{real,ai}.")
    ap.add_argument("--per_class", type=int, default=40000,
                    help="Max WildFake images to add per class (real / ai).")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of the added images routed to data/val instead of train.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delete_zips", action="store_true",
                    help="Delete the source zips after extraction to free disk.")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    dst = Path(a.dst).resolve()
    rng = random.Random(a.seed)

    for group, zips, cls in (("REAL", REAL_ZIPS, "real"), ("AI", AI_ZIPS, "ai")):
        print(f"\n=== {group} -> {dst}/train|val/{cls} ===")
        picked = _pick_members(src, zips, a.per_class, rng)
        if not picked:
            print("  nothing to do")
            continue
        n_ok, n_val, n_bad = _extract(
            picked, dst / "train" / cls, dst / "val" / cls, a.val_frac, rng)
        print(f"  added {n_ok} images ({n_val} to val, {n_bad} unreadable/skipped)")

    if a.delete_zips:
        print("\nDeleting source zips ...")
        for rel, _ in REAL_ZIPS + AI_ZIPS:
            p = src / rel
            if p.exists() and p.stat().st_size > 1024:
                p.unlink()
                print(f"  removed {rel}")

    # Report the resulting mix.
    print("\n=== resulting data/train counts ===")
    for cls in ("real", "ai"):
        d = dst / "train" / cls
        if not d.exists():
            continue
        total = sum(1 for _ in d.iterdir())
        wf = sum(1 for p in d.iterdir() if p.name.startswith("wf_"))
        print(f"  {cls}: {total} total  ({wf} WildFake, {total - wf} CIFAKE)")
    print("\nRetrain with the usual command, e.g.:")
    print("  python train.py --model hybrid_clip --device cuda --batch_size 16 "
          "--num_workers 2 --robust_eval_every 5 --no_prompt")


if __name__ == "__main__":
    main()
