import os
import torch

# General Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FLICKR_DIR = os.path.join(DATA_DIR, "flickr8k")
IMAGES_DIR = os.path.join(FLICKR_DIR, "Images")
CAPTIONS_FILE = os.path.join(FLICKR_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Model Architecture
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"  # Vision transformer backbone
TEXT_MODEL_NAME = "bert-base-uncased"            # BERT text encoder backbone
IMAGE_EMBEDDING_DIM = 768                        # ViT-B/32 output dimension
TEXT_EMBEDDING_DIM = 768                         # BERT-base output dimension
SHARED_PROJECTION_DIM = 256                      # Dimension of the shared space
MAX_SEQUENCE_LENGTH = 64                         # Max token length for captions

# Acceleration/Resource Settings
FREEZE_BACKBONES = True
CACHE_FEATURES = True
CACHE_DIR = os.path.join(DATA_DIR, "cached_features")

# Training Hyperparameters
EPOCHS = 30
BATCH_SIZE = 256        # Increased from 32 to make contrastive learning harder
LEARNING_RATE = 5e-5    # Decreased from 1e-4 for more stable projection learning
WEIGHT_DECAY = 1e-3
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

# Contrastive Loss Configuration
INITIAL_TEMPERATURE = 0.07                       # Initial tau parameter (1/e^t)
# Log temperature will be initialized to log(1/INITIAL_TEMPERATURE) = log(14.28) ~ 2.659
