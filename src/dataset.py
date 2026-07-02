import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from src import config

class CLIPDataset(Dataset):
    """
    PyTorch Dataset for Image-Text Contrastive Learning
    Loads and processes images and tokenizes captions using a BERT Tokenizer
    """
    def __init__(self, df, tokenizer, image_transform=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image']
        caption = row['caption']

        # Load image
        image_path = os.path.join(config.IMAGES_DIR, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback to a blank image if loading fails
            image = Image.new("RGB", (224, 224), (255, 255, 255))
            print(f"Warning: Failed to load image {image_path}, using blank image. Error: {e}")

        # Apply image transformations
        if self.image_transform:
            image = self.image_transform(image)
        else:
            # Default transformation matching CLIP's resizing structure
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet standards
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = default_transform(image)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=config.MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),      # Shape: [MAX_SEQUENCE_LENGTH]
            "attention_mask": tokens["attention_mask"].squeeze(0)  # Shape: [MAX_SEQUENCE_LENGTH]
        }



def get_transforms(split="train"):
    """
    Get torchvision transforms for images
    Applies standard augmentations to the train split to prevent overfitting
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



def get_dataloaders():
    """
    Reads the captions file partitions the dataset using the standard academic splits
    and returns DataLoader instances for training, validation and test sets
    """
    # Load captions
    if not os.path.exists(config.CAPTIONS_FILE):
        raise FileNotFoundError(
            f"Captions file not found at {config.CAPTIONS_FILE}. Please run download_dataset.py first."
        )
    
    df = pd.read_csv(config.CAPTIONS_FILE)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Check if standard split files exist
    train_txt = os.path.join(config.FLICKR_DIR, "Flickr_8k.trainImages.txt")
    val_txt = os.path.join(config.FLICKR_DIR, "Flickr_8k.devImages.txt")
    test_txt = os.path.join(config.FLICKR_DIR, "Flickr_8k.testImages.txt")
    
    if os.path.exists(train_txt) and os.path.exists(val_txt) and os.path.exists(test_txt):
        print("Using standard academic splits from Flickr8k txt files...")
        with open(train_txt, "r") as f:
            train_images = set(line.strip() for line in f if line.strip())
        with open(val_txt, "r") as f:
            val_images = set(line.strip() for line in f if line.strip())
        with open(test_txt, "r") as f:
            test_images = set(line.strip() for line in f if line.strip())
            
        train_df = df[df["image"].isin(train_images)]
        val_df = df[df["image"].isin(val_images)]
        test_df = df[df["image"].isin(test_images)]
    else:
        print("Standard split files not found. Creating random splits (80% Train, 10% Val, 10% Test)...")
        # Unique images
        unique_images = df["image"].unique()
        import numpy as np
        np.random.seed(config.SEED)
        np.random.shuffle(unique_images)
        
        n_total = len(unique_images)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        train_imgs = set(unique_images[:n_train])
        val_imgs = set(unique_images[n_train:n_train + n_val])
        test_imgs = set(unique_images[n_train + n_val:])
        
        train_df = df[df["image"].isin(train_imgs)]
        val_df = df[df["image"].isin(val_imgs)]
        test_df = df[df["image"].isin(test_imgs)]
        
    print(f"Dataset split sizes - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Datasets
    train_dataset = CLIPDataset(train_df, tokenizer, image_transform=get_transforms("train"))
    val_dataset = CLIPDataset(val_df, tokenizer, image_transform=get_transforms("val"))
    test_dataset = CLIPDataset(test_df, tokenizer, image_transform=get_transforms("test"))
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,      # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, tokenizer
