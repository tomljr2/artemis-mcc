import pyarrow.parquet as pq

from data.export_nanochat import write_shards


def read_texts(path):
    return pq.read_table(path).column("text").to_pylist()


def test_documents_are_split_into_numbered_shards_with_val_last(tmp_path):
    train = [f"train doc {i}\n\nwith a blank line" for i in range(5)]
    val = ["val doc 0", "val doc 1"]
    paths = write_shards(train, val, tmp_path, docs_per_shard=2)
    # nanochat sorts the shard names and holds out the last one for validation.
    assert [p.name for p in paths] == [f"shard_{i:05d}.parquet" for i in range(4)]
    assert sorted(tmp_path.iterdir()) == paths
    assert [t for p in paths[:-1] for t in read_texts(p)] == train
    assert read_texts(paths[-1]) == val  # whole documents, blank lines and all
