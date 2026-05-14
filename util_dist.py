#########
# Distributed Training Helper
#########
import os
import functools
import torch
from torch import distributed as dist
import pickle


def is_main_process():
    rank, _ = get_dist_info()
    return rank == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_dist(backend="nccl", **kwargs):
    rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    # dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:54411', world_size=world_size, rank=rank)

def setup_ddp(use_dist: bool):
    """Initialize (or skip) DDP. Returns: (is_dist, rank, world, local_rank, device)"""
    if not use_dist or not torch.cuda.is_available():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, dev

    # --- torchrun-provided envs ---
    rank       = int(os.environ["RANK"])
    world      = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])      # index on *this node*

    # Sanity before touching CUDA
    cvd  = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
    ngpu = torch.cuda.device_count()
    print(f"[rank {rank}] CVD={cvd} visible_gpus={ngpu} LOCAL_RANK={local_rank}", flush=True)
    assert ngpu > 0, "No GPUs visible to this process"
    assert 0 <= local_rank < ngpu, f"LOCAL_RANK {local_rank} out of range (visible_gpus={ngpu}, CVD={cvd})"

    # Set device FIRST, then init PG
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    if not dist.is_initialized():
        # Optional NCCL guards (harmless if not needed)
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker0")
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world)

    # Nice to have: print GPU name once per rank
    print(f"[rank {rank}] using cuda:{local_rank} -> {torch.cuda.get_device_name(device)}", flush=True)
    return True, rank, world, local_rank, device


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda")
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results