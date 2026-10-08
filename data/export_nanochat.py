"""Write our FineWeb-Edu documents in nanochat's format, to train nanochat on our data.

nanochat reads numbered parquet files with one document per row and holds out the last
file for validation. We write our exact train/val split that way, so its val score is
measured on the same documents as ours.

Run:  python -m data.export_nanochat
Then, from the nanochat folder, with NANOCHAT_BASE_DIR set to the folder printed below,
train its tokenizer and model as usual.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.prepare_fineweb_edu import RAW_PATH, load_documents, split_documents

# nanochat looks for its data in <NANOCHAT_BASE_DIR>/base_data_climbmix, whatever the data is.
BASE_DIR = Path("data/processed/nanochat_fineweb")
DOCS_PER_SHARD = 10_000


def write_shards(
    train: list[str], val: list[str], out_dir: Path, docs_per_shard: int = DOCS_PER_SHARD
) -> list[Path]:
    """Train documents in numbered shards, then all val documents in the last one."""
    groups = [train[i : i + docs_per_shard] for i in range(0, len(train), docs_per_shard)]
    groups.append(val)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, docs in enumerate(groups):
        path = out_dir / f"shard_{i:05d}.parquet"
        # Row groups of 1,024 documents, like nanochat's own files: it reads one at a time.
        pq.write_table(pa.table({"text": docs}), path, row_group_size=1024)
        paths.append(path)
    return paths


def main() -> None:
    if not RAW_PATH.exists():
        print(f"missing {RAW_PATH}: run python -m data.prepare_fineweb_edu first")
        return
    docs, _ = load_documents()
    train, val = split_documents(docs)
    paths = write_shards(train, val, BASE_DIR / "base_data_climbmix")
    print(f"train: {len(train):,} documents in {len(paths) - 1} shards")
    print(f"val: {len(val):,} documents in {paths[-1].name}")
    print(f"set NANOCHAT_BASE_DIR={BASE_DIR.resolve()}")


if __name__ == "__main__":
    main()
