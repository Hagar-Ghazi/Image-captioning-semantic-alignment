import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src import config
from src.dataset import get_dataloaders
from src.model import CLIPModel

def train_epoch(model, dataloader, optimizer, scaler, device, use_cached=False):
    """
    Runs one training epoch.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        if use_cached:
            image_feats, text_feats = batch
            image_feats = image_feats.to(device, non_blocking=True)
            text_feats = text_feats.to(device, non_blocking=True)
            
            # Suppress warning by using torch.amp.autocast
            device_type = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device == 'cuda')):
                loss, _ = model.forward_features(image_feats, text_feats)
        else:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            device_type = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device == 'cuda')):
                loss, _ = model(images, input_ids, attention_mask)
            
        if device == 'cuda' and torch.cuda.is_available() and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, use_cached=False):
    """
    Evaluates the model on validation set.
    """
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validation")
    
    for batch in progress_bar:
        if use_cached:
            image_feats, text_feats = batch
            image_feats = image_feats.to(device, non_blocking=True)
            text_feats = text_feats.to(device, non_blocking=True)
            
            device_type = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device == 'cuda')):
                loss, _ = model.forward_features(image_feats, text_feats)
        else:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            device_type = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device == 'cuda')):
                loss, _ = model(images, input_ids, attention_mask)
            
        total_loss += loss.item()
        progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def extract_and_save_features(model, loader, split_name, device):
    """
    Passes the dataset through the frozen encoders to extract representations.
    """
    model.eval()
    all_image_features = []
    all_text_features = []
    
    for batch in tqdm(loader, desc=f"Extracting {split_name} features"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Extract raw features from frozen backbones
        img_feats = model.image_encoder(images).cpu()
        txt_feats = model.text_encoder(input_ids, attention_mask).cpu()
        
        all_image_features.append(img_feats)
        all_text_features.append(txt_feats)
        
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    torch.save(all_image_features, os.path.join(config.CACHE_DIR, f"{split_name}_image_features.pt"))
    torch.save(all_text_features, os.path.join(config.CACHE_DIR, f"{split_name}_text_features.pt"))

def main():
    # Ensure checkpoint folder exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Set manual seed for reproducibility
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 1. Check Caching & Loading Data
    use_cached = getattr(config, "CACHE_FEATURES", False) and getattr(config, "FREEZE_BACKBONES", False)
    
    print("Loading data...")
    try:
        train_loader, val_loader, _, _ = get_dataloaders()
    except Exception as e:
        print(f"Error loading dataloaders: {e}")
        print("Please make sure the dataset is downloaded using download_dataset.py.")
        return
        
    # 2. Build Model
    print("Initializing CLIP model (ViT + BERT)...")
    model = CLIPModel().to(device)
    
    # Extract features if caching is enabled and cache files don't exist
    if use_cached:
        train_img_cache = os.path.join(config.CACHE_DIR, "train_image_features.pt")
        train_txt_cache = os.path.join(config.CACHE_DIR, "train_text_features.pt")
        val_img_cache = os.path.join(config.CACHE_DIR, "val_image_features.pt")
        val_txt_cache = os.path.join(config.CACHE_DIR, "val_text_features.pt")
        
        cache_exists = (os.path.exists(train_img_cache) and 
                        os.path.exists(train_txt_cache) and 
                        os.path.exists(val_img_cache) and 
                        os.path.exists(val_txt_cache))
                        
        if not cache_exists:
            print("\nCache files not found so Starting one-time feature extraction to cache")
            print("This will take ~3 minutes on CPU, but training will be 1000x faster afterwards!")
            extract_and_save_features(model, train_loader, "train", device)
            extract_and_save_features(model, val_loader, "val", device)
            print("Feature extraction completed successfully! Cache saved.")
            
        print("Loading pre-computed features from cache...")
        train_img_feats = torch.load(train_img_cache)
        train_txt_feats = torch.load(train_txt_cache)
        val_img_feats = torch.load(val_img_cache)
        val_txt_feats = torch.load(val_txt_cache)
        
        from torch.utils.data import TensorDataset, DataLoader
        train_feat_dataset = TensorDataset(train_img_feats, train_txt_feats)
        val_feat_dataset = TensorDataset(val_img_feats, val_txt_feats)
        
        train_loader = DataLoader(
            train_feat_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_feat_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False
        )
        print("Fast cached data loaders initialized!")
        
    # 3. Setup Optimizer, Scaler and Scheduler
    if getattr(config, "FREEZE_BACKBONES", False):
        print("Training Mode: Frozen Backbones (Optimization active on Projection Heads only)")
        optimizer_grouped_parameters = [
            {"params": model.image_projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": model.text_projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": [model.logit_scale], "lr": config.LEARNING_RATE}
        ]
    else:
        print("Training Mode: Full fine-tuning (Optimization active on all layers)")
        optimizer_grouped_parameters = [
            {"params": model.image_encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": model.text_encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": model.image_projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": model.text_projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": [model.logit_scale], "lr": config.LEARNING_RATE}
        ]
        
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=config.WEIGHT_DECAY)
    
    # Modern AMP API (suppresses warning)
    if device == 'cuda' and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=config.LR_SCHEDULER_FACTOR, 
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    # 4. Training Loop
    best_val_loss = float("inf")
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_cached=use_cached)
        val_loss = validate(model, val_loader, device, use_cached=use_cached)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch} finished - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_clip_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"New best model saved with Val Loss: {val_loss:.4f}")
            
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
