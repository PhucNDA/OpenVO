import os
from model import *
from dataset import *
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.optim as optim

import logging
from datetime import datetime

# Distributed training
from torch.nn.parallel import DistributedDataParallel
from util_dist import (
    get_dist_info,
    setup_ddp,
)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from pathlib import Path

from utils import read_config


'''
Reproducibility
'''
args = read_config()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# torch.backends.cudnn.benchmark = False # may reduce performance
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True) # may reduce performance

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

def is_main_process():
    return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0

def save_checkpoint(model, optimizer, epoch, step, loss, out_dir, fname_prefix="model",
                    logger=None):
    """
    Saver
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # unwrap DDP if needed
    to_save = model.module if hasattr(model, "module") else model

    # assemble payload
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "loss": float(loss),
        "model_state_dict": to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # filenames
    fn = os.path.join(out_dir, f"{fname_prefix}_ep-{epoch:03d}.pt")

    # atomic write
    torch.save(payload, fn)

    logger.info(f"Saved checkpoint: {fn} (epoch={epoch})")



def train(model, optimizer, scheduler, train_dl, logger, eval_ep=0):
    
    for ep in range(args.epochs):
        '''
        Training
        '''

        if args.dist: # Sampler re-seeded per epoch -> convergence
            train_dl.sampler.set_epoch(ep)

        st_t = time.time()
        model.train()
        train_mean_loss_p = 0.0 # (Translation + Deg Rotation)
        train_mean_loss_A = 0.0 # (Fisher)

        train_bar = tqdm(train_dl, disable=not is_main_process())
        for step, (x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs) in enumerate(train_bar):
            
            x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs = x.to('cuda'), y.to('cuda'), intrs.to('cuda'), intrs_map.to('cuda'), depth_map0.to('cuda'), depth_map1.to('cuda'), depth_3d.to('cuda'), time_freqs.to('cuda')
            
            optimizer.zero_grad()
            if args.dist:
                loss = model.module.step(x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs)
            else:
                loss_p, loss_A = model.step(x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs)
            # clip norm for multi-freq
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if args.dist and dist.is_initialized():
                # log loss_mean instead of per-rank loss
                loss_val = float(loss.detach().item())
                loss_tensor = torch.tensor([loss_val], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_mean = (loss_tensor / dist.get_world_size()).item()
            else:
                loss_mean_p = float(loss_p)
                loss_mean_A = float(loss_A)
            # check lr
            current_lr = optimizer.param_groups[0]["lr"]

            train_mean_loss_p += float(loss_mean_p)
            train_mean_loss_A += float(loss_mean_A)
            train_bar.set_postfix({"train_p_loss": train_mean_loss_p/(step+1), "train_A_loss": train_mean_loss_A/(step+1)})
            logger.info(f'{step}/{len(train_dl)} -------- Loss_P: {train_mean_loss_p/(step+1)} / CurBatch_Loss: {loss_p} ----------- lr={current_lr:.6g}\n')


        '''
        Update lr default: not set
        '''
        if args.step_scheduler:
            scheduler.step()

        '''
        Save model
        '''
        if args.dist and dist.is_initialized():
            dist.barrier()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=eval_ep,
            step=step,
            loss=train_mean_loss_p,
            out_dir=args.model_path,
            fname_prefix="model",
            logger=logger
        )               
        if args.dist and dist.is_initialized():
            dist.barrier() 
        eval_ep = eval_ep + 1
        
    return 0

if __name__ == '__main__':
    

    ################################################
    if args.dist: # Distributed training
        is_dist, rank, world, local_rank, device = setup_ddp(args.dist)

    # Configure the logging settings
    log_dir = args.model_path + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(
        log_dir,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="w"  # 'w' to overwrite each run, 'a' to append
    )
    # Quiet down noisy libs
    logging.getLogger("PIL").setLevel(logging.WARNING)                # or ERROR
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # extra safety
    logging.getLogger("timm").setLevel(logging.ERROR)                 # optional (your screenshot shows timm warnings)
    logger = logging.getLogger(__name__)

    REDACT = {"data_intrinsics", "training_data", "testing_data"}
    view = {k: ("<redacted>" if k in REDACT else v) for k, v in vars(args).items()}
    logger.info("Args:\n%s", json.dumps(view, indent=2, sort_keys=True, default=str))
    ################################################

    device = 'cuda'
    '''
    Init model, optimizer, scheduler
    '''
    if args.model_type == 'openvo':
        model = OpenVO(args).to(device)
    elif args.model_type == 'zvo_lite':
        model = ZVOModel(args).to(device)

    # ---- warmup for Lazy/meta params (your logic kept) ----
    from torch.nn.parameter import UninitializedParameter
    has_uninit = any(isinstance(p, UninitializedParameter) for p in model.parameters())

    if args.dist and has_uninit: # some module are Lazy so dummy forward to relax them
        model.eval()
        with torch.no_grad():
            tmp = get_data_info(args.training_data, args, mode='train')
            tmp = VisualOdometryDataset(args, tmp, (args.img_h, args.img_w), mode='train')

            # IMPORTANT: DistributedSampler so each rank sees different samples
            from torch.utils.data.distributed import DistributedSampler
            rr, world_size = get_dist_info()
            sampler = DistributedSampler(tmp, num_replicas=world_size, rank=rank, shuffle=True)

            train_bar = DataLoader(
                tmp,
                batch_size=1,
                shuffle=False,              # shuffle handled by sampler
                sampler=sampler,
                num_workers=2,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            for step, (x, y, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs) in enumerate(train_bar):
                x = x.to(device); y = y.to(device)
                intrs = intrs.to(device); intrs_map = intrs_map.to(device)
                depth_map0 = depth_map0.to(device); depth_map1 = depth_map1.to(device)
                depth_3d = depth_3d.to(device); time_freqs = time_freqs.to(device)
                _ = model(x, intrs, intrs_map, depth_map0, depth_map1, depth_3d, time_freqs)
                break
            x = None; y = None; intrs = None; intrs_map = None;
            depth_map0 = None; depth_map1 = None; depth_3d = None
            train_bar = None
            torch.cuda.empty_cache()
        model.train()
        torch.cuda.empty_cache()


        
    checkpoint = torch.load(os.path.join(args.pretrained_flownet_path, '8caNov12-1532_300000.pth'))
    pretrained_w = {}

    model_dict = model.state_dict()
    for key in checkpoint.keys():
        pretrained_w['encoder.maskflownet.'+key] = checkpoint[key]  
    pretrained_dict = {k: v for k, v in pretrained_w.items() if k in model_dict.keys()}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if args.dist:
        rrank, world_size = get_dist_info()
        total_batch_size = args.batch_size * world_size
        scaled_lr = args.learning_rate * (total_batch_size / args.batch_size)
        logger.info(f"Scale LR from {args.learning_rate} (batch size {args.batch_size}) to {scaled_lr} (batch size {total_batch_size})")
    else:
        scaled_lr = args.learning_rate
        logger.info(f"Scale LR from {args.learning_rate} (batch size {args.batch_size}) to {scaled_lr} (batch size {args.batch_size})")
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    
    if args.ssl_pretrain:
        print("###########")
        print("Load Pretrained Model!")
        print("###########")
        checkpoint = torch.load('########YOUR_PATH_HERE######')
        model.load_state_dict(checkpoint['model_state_dict'])
    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        eval_ep = checkpoint['epoch']
    else:
        eval_ep = 0

    checkpoint = None
    torch.cuda.empty_cache()
    
    if args.dist:
        # wrap with the **local_rank** as device id
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    '''
    Generate training dataloader
    '''
    train_df = get_data_info(args.training_data, args, mode='train')
    train_dataset = VisualOdometryDataset(args, train_df, (args.img_h, args.img_w), mode='train')

    if args.dist:
        sampler = DistributedSampler(train_dataset)
        train_dl = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.n_processors,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=sampler,
            )
    else:

        train_dl = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.n_processors,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    '''
    Train and Test
    '''

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train(model, optimizer, scheduler, train_dl, logger, eval_ep)
