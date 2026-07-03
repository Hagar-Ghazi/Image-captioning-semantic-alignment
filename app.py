import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import os
import torch

from src import config
from src.inference import (
    load_generative_model,
    generate_caption,
    load_custom_clip_model,
    compute_custom_similarities
)

# Page configuration
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and tab layouts
st.markdown("""
<style>
.stApp {
    background-color: #09090F;
    background-image: radial-gradient(circle at 50% 50%, #15122E 0%, #09090F 100%);
    color: #E2E8F0;
}
header, footer, .viewerBadge_container__17x7G {visibility: hidden; display: none !important;}

.title-container {
    text-align: center;
    padding: 1.5rem 0;
}
.app-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #C084FC 0%, #6366F1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Info Box */
.info-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.85rem;
    color: #94A3B8;
}
.info-box strong { color: #C084FC; }

/* Styled Button */
div.stButton > button {
    background: linear-gradient(135deg, #A855F7 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3) !important;
    width: 100% !important;
    cursor: pointer !important;
}

/* Output Card */
.output-card {
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 14px;
    padding: 1.2rem;
    margin-top: 1.2rem;
    text-align: center;
}
.output-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #C084FC;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.output-text {
    font-size: 1.3rem;
    font-weight: 600;
    color: #F8FAFC;
}

/* Styled text area */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADING -----------------
@st.cache_resource
def load_blip():
    return load_generative_model("Salesforce/blip-image-captioning-base", config.DEVICE)

@st.cache_resource
def load_custom():
    return load_custom_clip_model(config.DEVICE)

# Initialize models
blip_model, blip_processor = load_blip()
custom_model, custom_tokenizer, is_trained = load_custom()

# Title
st.markdown('<div class="title-container"><div class="app-title">📸 Image Captioning Portal</div></div>', unsafe_allow_html=True)

# Two Options Tabs
tab1, tab2 = st.tabs(["✍️ Generate Caption", "🔍 Find Best Caption"])

# --- TAB 1: GENERATE CAPTION ---
with tab1:
    st.markdown("""
    <div class="info-box">
        <strong>Model:</strong> BLIP (Generative) | 
        <strong>Input:</strong> Image Only | 
        <strong>Output:</strong> New text description
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload photo only:")
    uploaded_file1 = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploader1", label_visibility="collapsed")
    
    if uploaded_file1 is not None:
        img1 = Image.open(uploaded_file1)
        st.image(img1, use_container_width=True)
        
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if st.button("Generate Caption", key="btn1"):
            with st.spinner("Writing caption..."):
                caption = generate_caption(img1, blip_model, blip_processor, config.DEVICE)
                
            st.markdown(f"""
            <div class="output-card">
                <div class="output-header">🔮 Generated Description</div>
                <div class="output-text">"{caption}"</div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: FIND BEST CAPTION ---
with tab2:
    st.markdown("""
    <div class="info-box">
        <strong>Model:</strong> Custom CLIP (ViT + BERT) | 
        <strong>Input:</strong> Image + 5 Text options | 
        <strong>Output:</strong> Semantic match scores
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload image:")
    uploaded_file2 = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploader2", label_visibility="collapsed")
    
    default_captions = [
        "A cute dog playing happily in the green grass field.",
        "A group of people hiking up a rocky mountain peak.",
        "A professional office setting with developers writing code.",
        "A modern futuristic city street illuminated by neon lights.",
        "A close-up shot of a delicious hot pizza fresh from the oven."
    ]
    
    st.markdown("### Write 5 captions to choose between:")
    captions_text = st.text_area("Captions", value="\n".join(default_captions), height=140, label_visibility="collapsed")
    
    if uploaded_file2 is not None:
        img2 = Image.open(uploaded_file2)
        st.image(img2, use_container_width=True)
        
        candidate_captions = [line.strip() for line in captions_text.split("\n") if line.strip()]
        
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if st.button("Find Best Caption", key="btn2"):
            if len(candidate_captions) < 2:
                st.error("Please enter at least 2 captions.")
            else:
                with st.spinner("Aligning semantics..."):
                    probs, raw_sims = compute_custom_similarities(
                        img2, candidate_captions, custom_model, custom_tokenizer, config.DEVICE
                    )
                
                results_df = pd.DataFrame({
                    "Caption": candidate_captions,
                    "Confidence": probs,
                    "Similarity": raw_sims
                }).sort_values(by="Confidence", ascending=False)
                
                top_match = results_df.iloc[0]
                
                # Top match display
                st.markdown(f"""
                <div class="output-card" style="border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.05);">
                    <div class="output-header" style="color: #10B981;">🥇 Best Semantic Match</div>
                    <div class="output-text">"{top_match['Caption']}"</div>
                    <div style="font-size: 0.9rem; color: #10B981; margin-top: 0.5rem; font-weight: 700;">
                        Match Confidence: {top_match['Confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple Bar Chart
                chart = alt.Chart(results_df).mark_bar(cornerRadiusEnd=5).encode(
                    x=alt.X("Confidence:Q", title="", axis=alt.Axis(format='%')),
                    y=alt.Y("Caption:N", sort="-x", title=""),
                    color=alt.Color("Confidence:Q", scale=alt.Scale(scheme="purples"), legend=None)
                ).properties(height=180)
                
                st.altair_chart(chart, use_container_width=True)
