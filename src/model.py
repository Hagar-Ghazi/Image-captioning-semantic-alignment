import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, BertModel
from src import config

class VisionEncoder(nn.Module):
    """
    Encodes images using a pretrained CLIP Vision Transformer (ViT).
    """
    def __init__(self, model_name=config.IMAGE_MODEL_NAME):
        super().__init__()
        # Load only the vision portion of CLIP (ViT)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        # Freeze backbone parameters if configured to save CPU/GPU memory & compute
        freeze = getattr(config, "FREEZE_BACKBONES", False)
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # pooler_output represents the pooled representation of the [CLS] token of the image patches.
        # Shape: [batch_size, hidden_size] (768 for ViT-B/32)
        return outputs.pooler_output


class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained BERT model.
    """
    def __init__(self, model_name=config.TEXT_MODEL_NAME):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        freeze = getattr(config, "FREEZE_BACKBONES", False)
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation (first token) from the last hidden states as the sentence embedding.
        # Shape: [batch_size, hidden_size] (768 for BERT-base)
        return outputs.last_hidden_state[:, 0, :]


class CLIPModel(nn.Module):
    """
    Dual-Encoder Contrastive Text-Image Matching Model.
    Projects image and text embeddings into a shared space, L2-normalizes them,
    and computes the log-temperature scaled dot product InfoNCE loss.
    """
    def __init__(self, 
                 image_dim=config.IMAGE_EMBEDDING_DIM, 
                 text_dim=config.TEXT_EMBEDDING_DIM, 
                 projection_dim=config.SHARED_PROJECTION_DIM,
                 initial_temp=config.INITIAL_TEMPERATURE):
        super().__init__()
        self.image_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        
        # Projection heads: project raw representations to the shared embedding space (W_i and W_t)
        # Standard CLIP uses a linear projection without bias
        self.image_projection = nn.Linear(image_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        
        # Learnable log-temperature parameter `t`
        # In CLIP, logit scale is initialized to log(1 / initial_temp)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / initial_temp)))

    def get_image_embeddings(self, images):
        """Extracts and normalizes image embeddings."""
        image_features = self.image_encoder(images)
        image_projected = self.image_projection(image_features)
        # L2 Normalize: I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
        image_embeddings = F.normalize(image_projected, p=2, dim=-1)
        return image_embeddings

    def get_text_embeddings(self, input_ids, attention_mask):
        """Extracts and normalizes text embeddings."""
        text_features = self.text_encoder(input_ids, attention_mask)
        text_projected = self.text_projection(text_features)
        # L2 Normalize: T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
        text_embeddings = F.normalize(text_projected, p=2, dim=-1)
        return text_embeddings

    def forward(self, images, input_ids, attention_mask):
        # 1. Get projected and normalized embeddings
        I_e = self.get_image_embeddings(images)
        T_e = self.get_text_embeddings(input_ids, attention_mask)
        
        # 2. Compute logits using the exact dot product and log-temperature scaling formula:
        # logits = np.dot(I_e, T_e.T) * np.exp(t)
        # Logit scale is `t`, so torch.exp(self.logit_scale) is `np.exp(t)`
        t = self.logit_scale
        # Shape: [batch_size, batch_size]
        logits_per_image = torch.matmul(I_e, T_e.t()) * torch.exp(t)
        logits_per_text = logits_per_image.t()
        
        # 3. Compute symmetric InfoNCE loss
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=images.device)
        
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2
        
        return loss, logits_per_image

    def forward_features(self, image_features, text_features):
        """
        Forward pass using pre-computed backbone features instead of raw inputs.
        Extremely fast for CPU training with frozen backbones.
        """
        image_projected = self.image_projection(image_features)
        image_embeddings = F.normalize(image_projected, p=2, dim=-1)
        
        text_projected = self.text_projection(text_features)
        text_embeddings = F.normalize(text_projected, p=2, dim=-1)
        
        t = self.logit_scale
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) * torch.exp(t)
        logits_per_text = logits_per_image.t()
        
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2
        
        return loss, logits_per_image
