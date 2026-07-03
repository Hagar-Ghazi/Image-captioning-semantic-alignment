# 🔮 Professional Image Captioning and Cross-Modal Semantic Alignment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://image-captioning-semantic-alignment-icsrycmspweemhye9sqnpr.streamlit.app/)

A professional, high-performance vision-language system featuring a custom CLIP-style (ViT + BERT) contrastive alignment model and a state-of-the-art BLIP generative captioner built with an interactive dark-themed Streamlit web interface. 

The CLIP model is a "Dual-Encoder" consisting of two encoders (one for images, one for text) that map inputs to points in a shared vector space. It is designed to compare representations and calculate how semantically similar they are. It does not have a text decoder (like GPT-2) to generate new words. 

BLIP is a "Generative" model containing both an image encoder and a causal language decoder (a text generator) allowing it to produce and write new descriptive words token-by-token.

---

## 🖥️ App Screenshots

<p align="center">
  <img src="assets/App%20image1.png" width="48%" alt="App Screenshot 1" />
  <img src="assets/App%20image2.png" width="48%" alt="App Screenshot 2" />
</p>
<p align="center">
  <img src="assets/App%20image3.png" width="48%" alt="App Screenshot 3" />
  <img src="assets/App%20image4.png" width="48%" alt="App Screenshot 4" />
</p>
<p align="center">
  <img src="assets/App%20image5.png" width="97%" alt="App Screenshot 5" />
</p>

---

## Core Features

### Feature 1: Writing a new description from scratch
* **Model used:** BLIP (Pre-trained)
* **What it does:** When you upload an image and click "Generate Caption", the app uses the BLIP model to write a brand-new descriptive sentence from scratch.

### Feature 2: Matching and ranking candidate descriptions
* **Model used:** Our Custom CLIP Model (ViT + BERT)
* **What it does:** If you give the app an image and a list of 5 sentences, our custom model uses the temperature-scaled dot product to calculate the similarity percentages and rank which of the 5 sentences matches the image best.

---

## 📐 The Core Architecture

This project maps images and text captions into a shared semantic embedding space where matching pairs are mathematically aligned (close together) and mismatched pairs are pushed apart.

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

---

## 🧠 Step-by-Step Mathematical Explanation (Why and What)

To understand how the model aligns visual and textual concepts let's break down the role of each step in the pipeline:

### 1. Preprocessing and Encoding Branches

#### Image Branch:
* **Resize 224×224**: 
  * *Why:* Vision Transformers (ViT) are trained on fixed-resolution inputs. Resizing ensures images are structured into uniform spatial layouts for the patch embedder.
* **Patch Embedding**: 
  * *Why:* Unlike Convolutional Networks (CNNs) that scan pixels with sliding windows a Vision Transformer treats images like sentences. It cuts the 224x224 image into non-overlapping patches (e.g. 16x16 or 32x32 pixels) and projects them into flat vectors. These patches function exactly like words in a sentence.
* **CLIP Vision Transformer (ViT)**: 
  * *Why:* It applies self-attention mechanisms across all image patches to learn global context (how different parts of the image relate to each other). The final output (pooler_output) is a dense 768-dimensional vector representing the entire image's visual context.

#### Text Branch:
* **Tokenization**: 
  * *Why:* Raw text cannot be read by neural networks. The BERT tokenizer splits text into sub-word tokens (WordPiece) and maps them to numerical indices.
* **Word Embedding**: 
  * *Why:* Converts token indices into dense vectors representing individual word meanings.
* **BERT Encoder**: 
  * *Why:* BERT uses multi-head self-attention to read sentences bi-directionally meaning it learns the meaning of a word based on its surrounding context. We extract the representation of the special [CLS] token (768-dim) which acts as a summary embedding for the entire sentence.

---

### 2. Semantic Space Alignment (Linear Projection and Normalization)

* **Linear Projection (W_i, W_t)**:
  * *Why:* Even though the image embedding (from ViT) and text embedding (from BERT) are both 768-dimensional they reside in completely different vector spaces. Comparing them directly is like comparing apples to oranges.
  * We pass both through independent, learnable linear layers (projection heads) that map them into a shared space of dimension D (e.g., 256):
    I_p = I_f * W_i
    T_p = T_f * W_t
* **L2 Normalization**:
  * *Why:* The raw output vectors of projection layers can have arbitrary magnitudes (lengths). If we take their dot product directly, large vectors will dominate the similarity scores. 
  * We normalize the projected vectors to unit length (L2 norm = 1):
    I_e = I_p / ||I_p||_2
    T_e = T_p / ||T_p||_2
  * Now the dot product of these normalized vectors is mathematically equal to the Cosine Similarity bounding the scores between -1 (opposite) and +1 (identical).

---

### 3. 📉 Similarity and Loss (InfoNCE)

* **Dot Product and Similarity Matrix**:
  * *Why:* In a batch of size N, we compute the dot product between every image embedding and every text embedding. This produces an N x N similarity matrix representing all pairwise alignments:
    Similarities = I_e * T_e^T

* **Temperature Scaling (exp(t))**:
  * *Why:* Raw cosine similarities are very close to each other. When we pass them through a Softmax function to calculate classification probabilities, the resulting distribution is very flat (low confidence). 
  * To solve this, we scale similarities by a temperature factor 1/tau. In CLIP this is parameterized as a learnable log-temperature parameter t (where tau = 1/exp(t)):
    logits = (I_e * T_e^T) * exp(t)
  * Multiplying by exp(t) sharpens the Softmax distribution. During training, this penalizes wrong matches heavily and rewards correct matches with high confidence.

* **Softmax and Symmetric InfoNCE Loss**:
  * *Why:* We train the model by computing cross-entropy loss in two directions:
    1. Image-to-Text (Row-wise Softmax): For a given image, which caption in the batch matches it?
    2. Text-to-Image (Column-wise Softmax): For a given caption, which image in the batch matches it?
  * The final loss is the average of these two forcing matching pairs to have a similarity of +1 and mismatching pairs to have a similarity of 0 or less.

---

## How the Project Works (From Start to End)

The codebase supports two distinct operational modes:

### 1. Zero-Shot Retrieval Mode (Contrastive)
* Our Custom CLIP model is a "Dual-Encoder". It consists of two encoders (one for images, one for text) that map inputs to points in a shared vector space. It only knows how to compare two things and calculate similarity. It does not have a text decoder to generate new words.

* **Workflow:** You upload an image and input a list of custom candidate captions.
* **Under the Hood:**
  1. The image is passed through the preprocessed image pipeline and visual projection.
  2. The candidate sentences are tokenized, processed by BERT and passed through the text projection.
  3. The model computes the Cosine Similarity matrix and uses Softmax to output a probability distribution identifying which description best matches the image.
* **Primary Use Case:** Semantic text-image search and zero-shot classification.

### 2. Generative Captioning Mode (Generative)
* The pre-trained BLIP model contains both an image encoder and a causal language decoder. It generates new descriptive words from scratch token-by-token.

* **Workflow:** You upload an image and click "Generate Caption".
* **Under the Hood:**
  1. The image features are extracted using the Salesforce BLIP visual encoder.
  2. An autoregressive language decoder takes these image features as prompt context and generates a descriptive caption token-by-token.
* **Primary Use Case:** Producing detailed, object-aware natural language descriptions for any uploaded image from scratch.

---

## Step-by-Step Operations inside our Custom Model
1. **Extract Image Vector:** We pass the image through the Vision Transformer (ViT) backbone outputting a 768-dimensional vector representing visual features.
2. **Extract Text Vector:** We pass the caption through the BERT backbone, outputting a 768-dimensional vector representing text semantics.
3. **Project to Shared Space:** We pass both vectors through linear projection layers to map both down to a shared 256 dimensions.
4. **L2 Normalize:** We divide each vector by its length to make it a unit vector (length = 1).
5. **Calculate Similarity:** We compute the matrix dot product of the image and text vectors to obtain the Cosine Similarity.
6. **Scale by Temperature:** We multiply the similarity by exp(t) (our learnable temperature parameter).
7. **Compute Loss:** We calculate the cross-entropy loss in both directions (Image to Text and Text to Image) to train the projections.

---

## Repository Structure

```
Image Captioning Project/
├── assets/                       # App screenshots and media assets
├── data/                         # Datasets directory (e.g. Flickr8k)
│   └── cached_features/          # Pre-extracted visual/textual features
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, model versions, and paths
│   ├── dataset.py                # PyTorch Dataset loader & transform scripts
│   ├── model.py                  # CLIP ViT + BERT model with projection & loss
│   ├── train.py                  # High-speed feature-cached training loop
│   └── eval.py                   # Retrieval evaluation (Recall@1, 5, 10)
├── app.py                        # Streamlit web app (Clean dark UI)
├── download_dataset.py           # Automatic dataset downloader & formatting utility
├── requirements.txt              # Library dependencies
├── Dockerfile                    # Containerization script
└── README.md                     # Documentation
```

---

## Setup and Execution Guide

### 1. Installation
```bash
# Clone the repository
git clone 
cd "Image Captioning Project"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install libraries
pip install -r requirements.txt
```

### 2. High-Speed Training (CPU Optimized)
```bash
# 1. Download and set up Flickr8k dataset
python download_dataset.py

# 2. Run feature extraction and train projection heads
python -m src.train
```
*Note: Because the backbones are frozen and features are cached, the first run extracts features (~3 minutes) and the 30-epoch training loop finishes in under 15 seconds! We recommend running this on https://lightning.ai/ for optimal hosting performance.*

### 3. Run Evaluation
Verify the trained model's performance on the test set:
```bash
python -m src.eval
```

### 4. Run the Streamlit Portal
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser to interact with the model via a beautiful dark-mode interface
