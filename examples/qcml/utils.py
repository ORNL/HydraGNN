def balanced_block(rank: int, world_size: int, N: int):
    """
    Return [start, end) for this rank: consecutive indices start..end-1.
    Handles any N and world_size, including N < world_size.
    """
    base = N // world_size  # minimum chunk size
    extra = N % world_size  # number of ranks that get one extra item

    if rank < extra:
        start = rank * (base + 1)
        end = start + (base + 1)
    else:
        start = extra * (base + 1) + (rank - extra) * base
        end = start + base

    return start, end  # half-open interval
