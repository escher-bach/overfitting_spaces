from overfitting_spaces.data import make_split_manifest


def test_stratified_manifest_is_complete_and_deterministic():
    labels = [label for label in range(10) for _ in range(5_000)]
    left, right = make_split_manifest(labels), make_split_manifest(labels)
    assert left == right
    partitions = left["partitions"]
    assert sum(map(len, partitions.values())) == 50_000
    assert len(set().union(*map(set, partitions.values()))) == 50_000
    assert all(counts == {label: size // 10 for label in range(10)} for counts, size in ((left["class_counts"][name], len(indices)) for name, indices in partitions.items()))
