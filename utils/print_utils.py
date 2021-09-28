import torch.distributed as dist


def print_nothing(*args, **kwargs):
    pass


def print_master(*args, **kwargs):

    if not dist.is_initialized():
        print(*args, **kwargs)

    else:
        world_rank = dist.get_rank()
        if 0 == world_rank:
            print(*args, **kwargs)


def print_all_processes(*args, **kwargs):

    print(*args, **kwargs)


"""
Verbosity options for printing
0 - > nothing
1 -> master prints the basic
2 -> Master prints everything, progression bars included
3 -> all MPI processes print the basic 
4 -> all MPI processes print the basic, , progression bars included
"""

switcher = {
    0: print_nothing,
    1: print_master,
    2: print_master,
    3: print_all_processes,
    4: print_all_processes,
}


def print_distributed(verbosity_level, *args, **kwargs):

    print_verbose = switcher.get(verbosity_level)
    return print_verbose(*args, **kwargs)


def tqdm_verbosity_check(verbosity_level):
    return (0 == dist.get_rank() and 2 == verbosity_level) or 4 == verbosity_level
