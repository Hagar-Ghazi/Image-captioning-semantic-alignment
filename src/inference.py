import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel as HFCLIPModel,
    BertTokenizer
)
from src import config
from src.model import CLIPModel

# ----------------- BLIP Generative Captioner -----------------

def load_generative_model(model_name="Salesforce/blip-image-captioning-base", device=config.DEVICE):
    """
    Loads pretrained Salesforce BLIP model for image captioning.
    """
    print(f"Loading generative captioning model: {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, processor

def generate_caption(image, model, processor, device=config.DEVICE):
    """
    Generates a professional caption for an image using BLIP.
    """
    # Ensure image is in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate caption
    # Use beam search with nucleus sampling for descriptive and professional captions
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    # Capitalize the first letter
    return caption.capitalize()



# ----------------- Custom Model Similarity Matcher -----------------

def load_custom_clip_model(device=config.DEVICE):
    """
    Loads our custom CLIP model from checkpoints.
    If no checkpoint exists, returns the model with default (pretrained backbones but untrained projection) weights.
    """
    model = CLIPModel().to(device)
    tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_clip_model.pt")
    has_checkpoint = False
    
    if os.path.exists(checkpoint_path):
        print(f"Loading custom CLIP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        has_checkpoint = True
    else:
        print("No custom CLIP checkpoint found yet. Matcher will use pretrained backbones.")
        
    model.eval()
    return model, tokenizer, has_checkpoint

def compute_custom_similarities(image, texts, model, tokenizer, device=config.DEVICE):
    """
    Computes similarities between an image and a list of text captions using our custom model.
    Uses the exact formula: logits = dot(I_e, T_e.T) * exp(t) followed by softmax.
    """
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]
    
    # Prepare text
    tokens = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=config.MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        # Get embeddings
        I_e = model.get_image_embeddings(image_tensor)                      # [1, shared_dim]
        T_e = model.get_text_embeddings(input_ids, attention_mask)          # [num_texts, shared_dim]
        
        # Exact dot product and log-temperature formula from user's snippet
        # logits = np.dot(I_e, T_e.T) * np.exp(t)
        logits = torch.matmul(I_e, T_e.t()) * torch.exp(model.logit_scale)   # [1, num_texts]
        
        # Softmax to get probability distribution
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    # Return similarities as standard cosine similarity as well (without temperature scaling)
    with torch.no_grad():
        cosine_sim = torch.matmul(I_e, T_e.t()).squeeze(0).cpu().numpy()
        
    return probabilities, cosine_sim



# ----------------- Pre-trained HF CLIP Matcher (Fallback) -----------------

def load_pretrained_hf_clip(device=config.DEVICE):
    """
    Loads official pretrained OpenAI CLIP for high-quality out-of-the-box matching.
    """
    print("Loading pretrained HF CLIP...")
    model = HFCLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def compute_pretrained_similarities(image, texts, model, processor, device=config.DEVICE):
    """
    Computes similarities using official OpenAI CLIP.
    """
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image is similarity scaled by model's temperature
        logits = outputs.logits_per_image
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Compute raw cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        cosine_sim = torch.matmul(image_embeds, text_embeds.t()).squeeze(0).cpu().numpy()
        
    return probs, cosine_sim
