import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import wandb
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config as project_config
from src.distillation.dataset import CocoDetectionForDistill
from src.distillation.models import HuggingFaceTeacherWrapper

def setup_ddp():
    """
    Initializes the distributed process group.
    Backend is selected based on the operating system.
    """
    backend = 'gloo' if sys.platform == 'win32' else 'nccl'
    
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if int(os.environ["RANK"]) == 0:
            print(f"DDP Initialized with '{backend}' backend. World size: {os.environ['WORLD_SIZE']}.")
    else:
        print("WARNING: DDP environment variables not found. Running in single-process mode.")

def cleanup_ddp():
    """Destroys the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()

def main_training_function(rank, world_size, cfg):
    """The main training loop for knowledge distillation with a ConvNeXT teacher."""
    device = rank
    is_main_process = (rank == 0)
    
    hf_token = None
    if is_main_process:
        print(f"Starting distillation for ConvNeXt teacher on {world_size} GPU(s).")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"run_ddp_convnext_{timestamp}_lr{cfg['learning_rate']}_bs{cfg['batch_size_per_gpu']}"
        try:
            from huggingface_hub import login
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            wandb_key = os.getenv("WANDB_API_KEY")
            if hf_token: login(token=hf_token)
            if wandb_key: wandb.login(key=wandb_key)
            wandb.init(project=cfg["wandb_project"], config=cfg, name=run_name)
        except Exception as e:
            print(f"Could not log in, continuing without W&B. Error: {e}")

        Path(cfg["best_weights_filename"]).parent.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    teacher_model = HuggingFaceTeacherWrapper(cfg["teacher_hf_id"], token=hf_token).to(device)
    teacher_model.eval()

    if is_main_process:
        torch.hub.load(str(project_config.RTDETR_SOURCE_DIR), "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    if world_size > 1:
        dist.barrier() 

    student_hub_model = torch.hub.load(str(project_config.RTDETR_SOURCE_DIR), "rtdetrv2_l", source='local', pretrained=True, trust_repo=True)
    student_model = student_hub_model.model.to(device)

    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640).to(device)
        student_features_list = student_model.encoder(student_model.backbone(x))
        student_channels = [f.shape[1] for f in student_features_list]
    
    teacher_dims = teacher_model.feature_dims 
    projection_layers = nn.ModuleList([
        nn.Conv2d(student_channels[i], teacher_dims[i], kernel_size=1) for i in range(len(student_channels))
    ]).to(device)
    
    if world_size > 1:
        student_model = DDP(student_model, device_ids=[device], find_unused_parameters=True)
        projection_layers = DDP(projection_layers, device_ids=[device], find_unused_parameters=True)
    
    transforms = T.Compose([
        T.Resize((640, 640)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = CocoDetectionForDistill(root=cfg["train_images_dir"], ann_file=cfg["train_ann_file"], transforms=transforms)
    val_dataset = CocoDetectionForDistill(root=cfg["val_images_dir"], ann_file=cfg["val_ann_file"], transforms=transforms)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size_per_gpu"], num_workers=cfg["num_workers"], pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size_per_gpu"], num_workers=cfg["num_workers"], pin_memory=True, sampler=val_sampler)

    student_module = student_model.module if world_size > 1 else student_model
    projection_module = projection_layers.module if world_size > 1 else projection_layers
    params = list(student_module.backbone.parameters()) + list(student_module.encoder.parameters()) + list(projection_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['scheduler_factor'], patience=cfg['scheduler_patience'])

    if is_main_process and wandb.run: wandb.watch((student_model, projection_layers), log="all", log_freq=100)
    best_val_loss = float('inf')
    early_stopping_counter = 0
        
    for epoch in range(cfg["epochs"]):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        student_model.train(); projection_layers.train()
        total_train_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", disable=not is_main_process)
        for images, _ in train_iterator:
            images = images.to(device)
            with torch.no_grad():
                teacher_features_list = teacher_model(images)
            student_features_list = student_model.module.encoder(student_model.module.backbone(images)) if world_size > 1 else student_model.encoder(student_model.backbone(images))
            
            total_loss = 0
            for i in range(len(student_features_list)):
                projected_feat = projection_module[i](student_features_list[i])
                teacher_resized = F.interpolate(teacher_features_list[i], size=projected_feat.shape[-2:], mode="bilinear", align_corners=False)
                total_loss += criterion(projected_feat, teacher_resized)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
        
        if world_size > 1:
            train_loss_tensor = torch.tensor(total_train_loss).to(device); dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
        else:
            avg_train_loss = total_train_loss / len(train_loader)

        student_model.eval(); projection_layers.eval()
        total_val_loss = 0.0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]", disable=not is_main_process)
        with torch.no_grad():
            for images, _ in val_iterator:
                images = images.to(device)
                teacher_features_list = teacher_model(images)
                student_features_list = student_model.module.encoder(student_model.module.backbone(images)) if world_size > 1 else student_model.encoder(student_model.backbone(images))
                loss = 0
                for i in range(len(student_features_list)):
                    projected = projection_module[i](student_features_list[i])
                    teacher_resized = F.interpolate(teacher_features_list[i], size=projected.shape[-2:], mode="bilinear", align_corners=False)
                    loss += criterion(projected, teacher_resized)
                total_val_loss += loss.item()
                
        if world_size > 1:
            val_loss_tensor = torch.tensor(total_val_loss).to(device); dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        else:
            avg_val_loss = total_val_loss / len(val_loader)
        
        if is_main_process:
            duration = time.time() - start_time
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {duration:.2f}s")
            if wandb.run: wandb.log({"epoch": epoch + 1, "train/avg_loss": avg_train_loss, "val/avg_loss": avg_val_loss})
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; early_stopping_counter = 0
                print(f"Validation loss improved. Saving best model...")
                best_weights = {**student_module.backbone.state_dict(), **student_module.encoder.state_dict()}
                torch.save({'model': best_weights}, cfg["best_weights_filename"])
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. Counter: {early_stopping_counter}/{cfg['early_stopping_patience']}")

        if world_size > 1:
            stop_tensor = torch.tensor(1 if early_stopping_counter >= cfg['early_stopping_patience'] else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                if is_main_process: print("Early stopping triggered.")
                break
        elif early_stopping_counter >= cfg['early_stopping_patience']:
            print("Early stopping triggered.")
            break
            
    if is_main_process:
        print("\nDistillation finished.")
        if wandb.run: wandb.finish()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(project_config.ROOT_DIR / '.env')
    
    try:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            setup_ddp()

        cfg = {
            "learning_rate": 1e-4, "epochs": 50, "batch_size_per_gpu": 16, "num_workers": 2, "weight_decay": 1e-5,
            "teacher_hf_id": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            "train_images_dir": str(project_config.COCO_TRAIN_IMAGES), "val_images_dir": str(project_config.COCO_VAL_IMAGES),
            "train_ann_file": str(project_config.COCO_TRAIN_ANNOTATIONS), "val_ann_file": str(project_config.COCO_VAL_ANNOTATIONS),
            "scheduler_patience": 3, "scheduler_factor": 0.1, "early_stopping_patience": 7,
            "best_weights_filename": str(project_config.CONVNEXT_BEST_WEIGHTS),
            "wandb_project": project_config.WANDB_PROJECT_CONVNEXT_DISTILL,
        }
        main_training_function(rank, world_size, cfg)
    finally:
        cleanup_ddp()