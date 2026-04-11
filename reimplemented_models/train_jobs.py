"""Chia danh sách (dataset, arch, model_type) cho train song song nhiều GPU."""


def sharded_jobs(
    datasets: list[str],
    architectures: list[str],
    model_types: list[str],
    shard_index: int,
    shard_count: int,
) -> list[tuple[str, str, str]]:
    jobs = [(ds, arch, mt) for ds in datasets for arch in architectures for mt in model_types]
    if shard_count < 1:
        raise ValueError("shard_count phải >= 1")
    if shard_count == 1:
        return jobs
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"shard_index phải thuộc [0, {shard_count})")
    return [j for i, j in enumerate(jobs) if i % shard_count == shard_index]
