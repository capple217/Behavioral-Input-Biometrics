"""Tiny process-pool helper: per-subject work is independent and CPU-bound."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor


def _init():
    import torch

    torch.set_num_threads(1)


def pmap(fn, items, workers: int | None = None):
    workers = workers or int(os.environ.get("BIB_WORKERS", os.cpu_count() or 1))
    if workers <= 1:
        _init()
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        return list(ex.map(fn, items, chunksize=1))
