#!/usr/bin/env python3
"""Day-21: Fetch the 50 images needed for the human-validation pass by reading
parquet shards directly with pyarrow.

We deliberately bypass datasets.load_dataset() because MTVQA is sharded into
7 parquet files (~500 MB each) and load_dataset() blocks until every shard
is downloaded. Our 50 image_uids all map to row_indices in [0, 7000), which
fall inside the first 2 shards -- already cached -- so direct parquet access
avoids fetching the remaining 5 shards.

The image_uid encoding (see scripts/prepare_mtvqa.py:130):

  image_uid = f"{split}__{row_index:05d}__{source_image_id}"

where row_index is the global index into the concatenated MTVQA `split` (here
"train"), source_image_id is the dataset's `id` field.

Output:
  results/day18_human_validation/images/{anno_idx:02d}_{image_uid}.png
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "results/day18_human_validation/sample.jsonl"
OUT_DIR = REPO / "results/day18_human_validation/images"
HUB = Path.home() / ".cache/huggingface/hub/datasets--ByteDance--MTVQA"

UID_RE = re.compile(r"^(?P<split>train|val|test)__(?P<idx>\d{5})__(?P<src>.+)$")


def find_parquet_shards(split: str) -> list[Path]:
    snapshots = HUB / "snapshots"
    if not snapshots.exists():
        raise SystemExit(f"HF MTVQA snapshot dir missing: {snapshots}\n"
                         "Run a `datasets.load_dataset('ByteDance/MTVQA', split='train')` "
                         "once to populate the cache (it can be killed after the first 2 "
                         "shards finish downloading -- we only need those).")
    # take the first snapshot dir
    snap = next(snapshots.iterdir())
    data_dir = snap / "data"
    shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards for split={split} under {data_dir}")
    return shards


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with SAMPLE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[load] {len(rows)} sample rows", flush=True)

    needed: dict[str, list[tuple[int, str, int]]] = {}
    for r in rows:
        uid = r["image_uid"]
        m = UID_RE.match(uid)
        if not m:
            raise ValueError(f"unrecognised image_uid: {uid}")
        split = m.group("split")
        idx = int(m.group("idx"))
        anno_idx = r["_anno_idx"]
        needed.setdefault(split, []).append((idx, uid, anno_idx))

    fetched = 0
    for split, items in needed.items():
        print(f"\n[split={split}] need {len(items)} images", flush=True)
        shards = find_parquet_shards(split)
        print(f"[split={split}] found {len(shards)} shards on disk:", flush=True)
        # We need cumulative row counts per shard to map global row_index ->
        # (shard, local_index). Read metadata only -- this is cheap.
        cum = [0]
        for sh in shards:
            if not sh.exists():
                print(f"  [skip] {sh.name} (missing)", flush=True)
                cum.append(cum[-1])
                continue
            meta = pq.read_metadata(str(sh))
            n = meta.num_rows
            print(f"  {sh.name}: {n} rows  (global [{cum[-1]}, {cum[-1]+n}))", flush=True)
            cum.append(cum[-1] + n)

        max_global = cum[-1]
        items_sorted = sorted(items, key=lambda t: t[0])

        # Group by shard to avoid re-opening
        by_shard: dict[int, list[tuple[int, int, str, int]]] = {}  # shard_idx -> [(local_idx, global_idx, uid, anno_idx)]
        for gidx, uid, anno_idx in items_sorted:
            if gidx >= max_global:
                print(f"  [skip] global idx={gidx} (uid={uid}) is past available shards "
                      f"(max={max_global}); shard not downloaded yet", flush=True)
                continue
            # find which shard this index falls into
            shard_idx = 0
            while shard_idx + 1 < len(cum) and cum[shard_idx + 1] <= gidx:
                shard_idx += 1
            local_idx = gidx - cum[shard_idx]
            by_shard.setdefault(shard_idx, []).append((local_idx, gidx, uid, anno_idx))

        for shard_idx, group in sorted(by_shard.items()):
            sh = shards[shard_idx]
            print(f"\n[shard {shard_idx}] reading {sh.name}, picking {len(group)} rows ...", flush=True)
            # Read only the rows we need (and only the columns we need)
            local_indices = sorted(li for li, *_ in group)
            table = pq.read_table(str(sh), columns=["image", "id", "lang"])
            # Build a small lookup by local_index
            id_col = table.column("id").to_pylist()
            lang_col = table.column("lang").to_pylist()
            image_col = table.column("image")
            for local_idx, gidx, uid, anno_idx in sorted(group):
                src_id = id_col[local_idx]
                lang = (lang_col[local_idx] or "").lower()
                uid_check = f"{split}__{gidx:05d}__{src_id}"
                if uid_check != uid:
                    print(f"  [warn] uid mismatch at gidx={gidx}: expected={uid} "
                          f"got={uid_check}; using expected uid", flush=True)
                dst = OUT_DIR / f"{anno_idx:02d}_{uid}.png"
                if dst.exists():
                    print(f"  [skip] anno={anno_idx:02d} already on disk", flush=True)
                    continue
                # Image column is typically a struct with 'bytes' or already a PIL-decoded dict
                cell = image_col[local_idx].as_py()
                if isinstance(cell, dict) and "bytes" in cell and cell["bytes"]:
                    img = Image.open(io.BytesIO(cell["bytes"]))
                elif isinstance(cell, dict) and "path" in cell and cell["path"]:
                    img = Image.open(cell["path"])
                else:
                    print(f"  [error] unknown image format at gidx={gidx}: {type(cell).__name__}", flush=True)
                    continue
                img.convert("RGB").save(dst)
                fetched += 1
                print(f"  [save] anno={anno_idx:02d} gidx={gidx} lang={lang} -> {dst.name}", flush=True)

    have = sum(1 for f in OUT_DIR.iterdir() if f.suffix == ".png")
    print(f"\n[done] fetched={fetched}, total on disk={have} (target=50)", flush=True)


if __name__ == "__main__":
    main()
