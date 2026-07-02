# Professional Image Captioning & Text-Image Semantic Matching

This project implements a professional dual-encoder contrastive text-image alignment model (inspired by OpenAI's **CLIP** architecture) using a **Vision Transformer (ViT)** for images and a **BERT Encoder** for text. It also features a state-of-the-art **Salesforce BLIP** generative captioner and a modern **Streamlit Web Portal** for interactive image analysis.

---

## 🔮 Architecture Overview

The system processes images and text captions concurrently, mapping them to a joint embedding space where corresponding pairs are close and mismatched pairs are far:

```
            IMAGE                             TEXT
              │                                 │
      Resize 224×224                       Tokenization
              │                                 │
      Patch Embedding                     Word Embedding
              │                                 │
  CLIP Vision Transformer                  BERT Encoder
   (pooler_output: 768)                (CLS representation: 768)
              │                                 │
      Image Feature Vector              Text Feature Vector
              │                                 │
      Linear Projection                 Linear Projection
     (image_projection: 256)           (text_projection: 256)
              │                                 │
        L2 Normalize                      L2 Normalize
       (I_e: [N, 256])                   (T_e: [N, 256])
              │                                 │
              └─────────── Dot Product ─────────┘
                            │
                     Similarity Matrix [N, N]
                            │
                Temperature Scaling (exp(t))
                            │
                         Softmax
                            │
                 Symmetric InfoNCE Loss
```

### Core Equations
1. **L2 Normalization**:
   $$I_e = \frac{I_f W_i}{\|I_f W_i\|_2}$$
   $$T_e = \frac{T_f W_t}{\|T_f W_t\|_2}$$

2. **Similarity Logits**:
   $$\text{logits} = (I_e \cdot T_e^T) \cdot e^t$$
   Where $t$ is a learnable log-temperature parameter.

3. **Symmetric Loss**:
   $$\mathcal{L} = \frac{\mathcal{L}_{\text{I2T}} + \mathcal{L}_{\text{T2I}}}{2}$$
   $$\mathcal{L}_{\text{I2T}} = \text{CrossEntropy}(\text{logits}, \text{labels})$$
   $$\mathcal{L}_{\text{T2I}} = \text{CrossEntropy}(\text{logits}^T, \text{labels})$$

---

## 📁 Repository Structure

```
Image Captioning CV/
├── data/                         # Datasets directory
│   └── flickr8k/                 # Prepared Flickr8k dataset
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, model versions, and paths
│   ├── dataset.py                # PyTorch Dataset loader & transform scripts
│   ├── model.py                  # CLIP ViT + BERT model with exact logits formula
│   ├── train.py                  # Mixed precision training & checkpointing loop
│   ├── eval.py                   # Retrieval evaluation (Recall@1, Recall@5, Recall@10)
│   └── inference.py              # Inference helpers (BLIP captioner & custom model)
├── app.py                        # Streamlit web app
├── download_dataset.py           # Automatic dataset downloader & formatting utility
├── requirements.txt              # Project library dependencies
└── README.md                     # Documentation
```

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed. Set up a virtual environment and install dependencies:
```bash
# Clone the repository and navigate into it
cd "Image Captioning CV"

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Unix/macOS

# Install libraries
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download and prepare the Flickr8k dataset automatically:
```bash
python download_dataset.py
```
This utility will download the dataset (~1GB), extract the images to `data/flickr8k/Images`, and parse the raw captions file into a clean CSV format at `data/flickr8k/captions.txt`.

### 3. Model Training
Train the custom CLIP model (ViT + BERT) on the dataset:
```bash
python -m src.train
```
* Custom linear projection heads and learnable temperature scale are optimized alongside the pre-trained backbones.
* Uses **Automatic Mixed Precision (AMP)** for accelerated training.
* Checkpoints are automatically saved to `checkpoints/best_clip_model.pt` based on validation loss.

### 4. Retrieval Evaluation
Evaluate the custom model's retrieval capability on the unseen test set using standard **Recall@K** metrics:
```bash
python -m src.eval
```
This prints the Recall@1, Recall@5, and Recall@10 metrics for both Image-to-Text (I2T) and Text-to-Image (T2I) search queries.

---

## 🖥️ Running the Streamlit Web Application

To launch the premium web interface:
```bash
streamlit run app.py
```

### Web App Features:
1. **Interactive Image Uploader**: Upload any image.
2. **Generative Captioner (SOTA BLIP)**: Generates highly detailed and professional sentences describing all objects and actions inside the image.
3. **Cross-Modal Semantic Alignment**: Input custom text descriptions (e.g. *"a dog playing on grass"*, *"a cat on a red couch"*). The web portal ranks them using either official OpenAI CLIP or your custom-trained model, displaying alignment scores in a sleek bar chart.
4. **Visual Dashboard**: Displays matching percentages and detailed tabular metrics.
