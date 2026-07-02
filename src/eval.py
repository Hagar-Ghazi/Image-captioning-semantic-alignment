import os
import torch
import numpy as np
from tqdm import tqdm
from src import config
from src.dataset import get_dataloaders
from src.model import CLIPModel

@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extracts all normalized image and text embeddings for the entire dataset split
    Tracks mapping from captions to images
    """
    model.eval()
    all_image_embs = []
    all_text_embs = []
    
    # We want to get unique images and all captions.
    # Flickr8k has 5 captions per image. The dataloader gives us (image, caption) pairs.
    # We will compute embeddings for all pairs.
    image_names = []
    
    # To avoid computing embeddings for duplicate images, we keep track of unique images.
    unique_image_embs = {}
    
    print("Extracting embeddings for evaluation...")
    for idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Get embeddings
        img_embs = model.get_image_embeddings(images)
        txt_embs = model.get_text_embeddings(input_ids, attention_mask)
        
        # Save text embeddings
        all_text_embs.append(txt_embs.cpu())
        
        # Retrieve image names for this batch from df
        batch_size = images.size(0)
        start_idx = idx * dataloader.batch_size
        batch_df = dataloader.dataset.df.iloc[start_idx : start_idx + batch_size]
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_name = row['image']
            image_names.append(img_name)
            
            # Cache unique image embeddings
            if img_name not in unique_image_embs:
                unique_image_embs[img_name] = img_embs[i].cpu()
                
    # Stack text embeddings
    all_text_embs = torch.cat(all_text_embs, dim=0) # [Num_captions, shared_dim]
    
    # Stack unique image embeddings in order of appearance
    unique_image_names = list(unique_image_embs.keys())
    all_image_embs = torch.stack([unique_image_embs[name] for name in unique_image_names], dim=0) # [Num_unique_images, shared_dim]
    
    # Create mapping from text index to unique image index
    text_to_image_map = []
    img_name_to_idx = {name: idx for idx, name in enumerate(unique_image_names)}
    for name in image_names:
        text_to_image_map.append(img_name_to_idx[name])
    text_to_image_map = np.array(text_to_image_map) # [Num_captions]
    
    return all_image_embs, all_text_embs, text_to_image_map


def compute_recall(image_embeddings, text_embeddings, text_to_image_map):
    """
    Computes Recall@1, Recall@5, and Recall@10 for both Image-to-Text and Text-to-Image.
    """
    # L2-normalized embeddings, so dot product is Cosine Similarity
    # Similarity matrix: [Num_images, Num_captions]
    similarity = torch.matmul(image_embeddings, text_embeddings.t()).numpy()
    
    num_images = similarity.shape[0]
    num_captions = similarity.shape[1]
    
    # --- 1. Image-to-Text Retrieval (I2T) ---
    # For each image, retrieve top captions. Ground truths are captions where text_to_image_map == image_index
    i2t_recalls = {1: 0, 5: 0, 10: 0}
    
    for img_idx in range(num_images):
        # Ground truth caption indices for this image
        gt_captions = np.where(text_to_image_map == img_idx)[0]
        
        # Similarity scores for this image against all captions
        scores = similarity[img_idx]
        
        # Sort indices in descending order of similarity
        sorted_indices = np.argsort(scores)[::-1]
        
        # Check Recall@K
        for k in [1, 5, 10]:
            top_k_indices = sorted_indices[:k]
            # If any ground truth caption is in the top-k retrieved captions
            if any(caption in top_k_indices for caption in gt_captions):
                i2t_recalls[k] += 1
                
    for k in i2t_recalls:
        i2t_recalls[k] = (i2t_recalls[k] / num_images) * 100

    # --- 2. Text-to-Image Retrieval (T2I) ---
    # For each caption, retrieve top images. Ground truth is text_to_image_map[caption_index]
    t2i_recalls = {1: 0, 5: 0, 10: 0}
    
    for cap_idx in range(num_captions):
        # Ground truth image index for this caption
        gt_image = text_to_image_map[cap_idx]
        
        # Similarity scores for this caption against all images
        scores = similarity[:, cap_idx]
        
        # Sort image indices in descending order
        sorted_indices = np.argsort(scores)[::-1]
        
        # Check Recall@K
        for k in [1, 5, 10]:
            top_k_indices = sorted_indices[:k]
            if gt_image in top_k_indices:
                t2i_recalls[k] += 1
                
    for k in t2i_recalls:
        t2i_recalls[k] = (t2i_recalls[k] / num_captions) * 100
        
    return i2t_recalls, t2i_recalls

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, _, test_loader, _ = get_dataloaders()
    
    # 2. Build Model
    model = CLIPModel().to(device)
    
    # 3. Load Checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_clip_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")
    else:
        print("No checkpoint found. Running evaluation with randomly initialized model.")
        
    # 4. Extract Embeddings
    image_embs, text_embs, text_to_image_map = extract_embeddings(model, test_loader, device)
    
    # 5. Compute Recall
    print("Computing recalls...")
    i2t, t2i = compute_recall(image_embs, text_embs, text_to_image_map)
    
    # 6. Display results
    print("\n" + "="*30)
    print("        RETRIEVAL METRICS        ")
    print("="*30)
    print("Image-to-Text (I2T) Retrieval:")
    print(f"  Recall@1 : {i2t[1]:.2f}%")
    print(f"  Recall@5 : {i2t[5]:.2f}%")
    print(f"  Recall@10: {i2t[10]:.2f}%")
    print("-"*30)
    print("Text-to-Image (T2I) Retrieval:")
    print(f"  Recall@1 : {t2i[1]:.2f}%")
    print(f"  Recall@5 : {t2i[5]:.2f}%")
    print(f"  Recall@10: {t2i[10]:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()
